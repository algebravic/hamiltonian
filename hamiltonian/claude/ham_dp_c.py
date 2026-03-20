"""
ham_dp_c.py  (EH edition)
--------------------------
C-accelerated frontier DP for counting Hamiltonian paths in G_n.

Hash table redesign: Extendible Hashing (Fagin et al. 1979)
------------------------------------------------------------
Previous versions used a linear-probing hash table pre-allocated at 1.5×
the input state count (slot_cap-bounded).  This caused two problems:

  1. Pre-allocation mismatch: nxt was sized for 1.5× si even when the
     actual output was much smaller (e.g. step 40 n=61: 19.2M → 4.2M
     states, 6.2% fill, yet a 1.61 GB table was allocated).

  2. 3-table resize spike: when LP's nxt grew past the load factor during
     a sweep, it doubled (old_nxt + new_nxt + curr = 3 tables simultaneously),
     causing a transient 2–3× memory spike that pushed macOS into swap.

Extendible hashing fixes both:

  - Memory ∝ actual entries, not estimated capacity.
  - No resize spike: bucket splits are incremental (one new bucket per split).
  - Write buffer is no longer needed: EH already groups entries by hash prefix.
  - --load-factor tuning is no longer needed.

EH structure
------------
  EHBkt: 128 key/value pairs + cnt + local_depth = 3080 bytes.
         128 × 24B = 3072B of data fits within ~2 L1 cache lines for scan.

  EHSlab: 4096 contiguous EHBkt objects (~12.6 MB), bump-allocated.
          Pool is rewound (not freed) between steps.

  EHT: directory of 2^gd bucket pointers.
       Starts at gd=4 (16 root buckets).
       Doubles the directory when a bucket at full local depth fills.

  eh_insert: search bucket (L1-resident) for existing key → accumulate;
             otherwise insert; if bucket full → split + retry.
  eh_reset:  rewind pool (O(n_slabs)), rebuild tiny directory (O(16)).
  EH_FOR:    iterate unique buckets via pool traversal (not via directory).

Public API
----------
count_hamiltonian_paths_c(n, order, adj)  ->  int
_get_lib()                                ->  (lib, ffi)
"""

import os, hashlib, subprocess, tempfile, cffi

C_SOURCE = r"""
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <stdio.h>
#include <time.h>

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

typedef unsigned __int128 u128;
typedef uint64_t          u64;

#define KEY_MARKER  ((u64)1 << 63)
#define KEY_NC_BIT  ((u64)1 << 62)
/* 4 bits per frontier slot + 2 flag bits → max 15 slots in 64 bits. */
#define MAX_FS_FAST 15

/* ======================================================================
   State encoding helpers (unchanged from LP version)
   ====================================================================== */

static inline uint8_t enc_v(int8_t v)  { return (uint8_t)(v + 2); }
static inline int8_t  dec_v(uint8_t n) { return (int8_t) (n - 2); }

static inline int8_t slot_get(u64 key, int i)
{
    return dec_v((uint8_t)((key >> (4*i)) & 0xFu));
}
static inline u64 slot_set(u64 key, int i, int8_t v)
{
    key &= ~((u64)0xFu << (4*i));
    key |=  (u64)enc_v(v) << (4*i);
    return key;
}
static inline int nc_get(u64 key)   { return (int)((key >> 62) & 1u); }
static inline u64 nc_set_1(u64 key) { return key | KEY_NC_BIT; }

static inline int label_count(u64 key, int fs, int8_t lbl)
{
    uint8_t t = enc_v(lbl);
    int c = 0;
    for (int i = 0; i < fs; i++)
        if (((key >> (4*i)) & 0xFu) == t) c++;
    return c;
}
static inline int8_t label_max(u64 key, int fs)
{
    int8_t mx = 0;
    for (int i = 0; i < fs; i++) {
        int8_t v = dec_v((uint8_t)((key >> (4*i)) & 0xFu));
        if (v > mx) mx = v;
    }
    return mx;
}
static inline u64 label_rename(u64 key, int fs, int8_t old_lbl, int8_t new_lbl)
{
    uint8_t oe = enc_v(old_lbl), ne = enc_v(new_lbl);
    for (int i = 0; i < fs; i++) {
        if (((key >> (4*i)) & 0xFu) == oe) {
            key &= ~((u64)0xFu << (4*i));
            key |=  (u64)ne   << (4*i);
        }
    }
    return key;
}
static inline u64 canon(u64 key, int fs)
{
    uint8_t map[16];
    memset(map, 0, 16);
    uint8_t nxt = enc_v(1);
    u64 result = key & (KEY_MARKER | KEY_NC_BIT);
    for (int i = 0; i < fs; i++) {
        uint8_t n = (uint8_t)((key >> (4*i)) & 0xFu);
        if (dec_v(n) > 0) {
            if (!map[n]) map[n] = nxt++;
            n = map[n];
        }
        result |= (u64)n << (4*i);
    }
    return result;
}
static inline u64 introduce(u64 key, int fs)
{
    return key | ((u64)enc_v(0) << (4*fs));
}
static inline u64 eliminate_slot(u64 key, int u_idx, int fs)
{
    u64 low  = key & (((u64)1 << (4*u_idx)) - 1u);
    u64 high = (key >> (4*(u_idx+1)))
               & (((u64)1 << (4*(fs - u_idx - 1))) - 1u);
    return (key & (KEY_MARKER | KEY_NC_BIT)) | low | (high << (4*u_idx));
}

/* ======================================================================
   Extendible Hash Table: u64 key -> u128 value (accumulating +=)

   Bucket layout (3080 bytes):
     u64  k[128]  — 1024 bytes, 16-byte aligned after u128 arrays below
     u128 v[128]  — 2048 bytes (u128 requires 16-byte alignment)
     int  cnt     — occupied entries (0..128)
     int  ld      — local depth

   Pool (bump allocator):
     EHSlab holds EH_SLAB_N pre-allocated buckets.
     Slabs are appended when full and never freed until eh_free().
     eh_reset() rewinds pool without freeing: O(n_slabs), typically < 1 ms.

   Directory:
     dir[i] → bucket for keys with top-gd bits of eh_hash(key) == i.
     Starts at gd=4 (16 entries).  Doubles lazily during splits.
   ====================================================================== */

#ifndef EH_BKT_CAP
#define EH_BKT_CAP    32     /* entries per bucket: tuned at compile time    */
#endif
                             /* default 32; overridden via -DEH_BKT_CAP=N   */
#define EH_INIT_GD     4     /* initial global depth → 16 root buckets       */
#define EH_SLAB_N   4096     /* buckets per slab                             */

typedef struct EHBkt {
    u64  k[EH_BKT_CAP];     /* keys:   1024 bytes                           */
    u128 v[EH_BKT_CAP];     /* values: 2048 bytes (u128 aligned to 16)      */
    int  cnt;                /* number of occupied entries                   */
    int  ld;                 /* local depth                                  */
} EHBkt;                     /* sizeof = 3080 bytes                         */

typedef struct EHSlab {
    EHBkt        *bkts;      /* malloc'd array of EH_SLAB_N EHBkt objects   */
    int           used;      /* buckets consumed from this slab              */
    struct EHSlab *next;     /* linked list of slabs                         */
} EHSlab;

typedef struct {
    EHBkt       **dir;       /* directory: dir[i] → bucket for hash prefix i */
    int           gd;        /* global depth (dir has 2^gd entries)           */
    size_t        cnt;       /* total entry count across all buckets          */
    EHSlab       *sl_head;   /* first slab in pool chain                      */
    EHSlab       *sl_cur;    /* slab currently being filled                   */
} EHT;

/* Full-width 64-bit hash (same mix constants as LP version). */
static inline u64 eh_hash(u64 k) {
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

/* Allocate one bucket from the pool.  cnt is set to 0; caller sets ld.
   k[] and v[] may contain stale data from a previous step but are safe
   because they are only read/written within [0, cnt).
   If the current slab's bkts array was freed by eh_reset, re-allocate it
   now — this ensures the OS actually reclaims pages between steps.        */
static EHBkt *eh_bkt_alloc(EHT *t) {
    if (!t->sl_cur || t->sl_cur->used >= EH_SLAB_N) {
        /* Need a new (or recycled) slab.  Walk forward to find one whose
           bkts may already be allocated, or create a fresh slab.          */
        EHSlab *next = t->sl_cur ? t->sl_cur->next : NULL;
        if (next) {
            t->sl_cur = next;
        } else {
            EHSlab *sl = (EHSlab *)malloc(sizeof(EHSlab));
            sl->bkts   = NULL;   /* allocated on demand below              */
            sl->used   = 0;
            sl->next   = NULL;
            if (!t->sl_head) t->sl_head = sl;
            else             t->sl_cur->next = sl;
            t->sl_cur = sl;
        }
    }
    /* Re-allocate bkts if freed by eh_reset.  Large mallocs on macOS use
       mmap internally, so a matching free() actually unmaps the pages.    */
    if (!t->sl_cur->bkts) {
        t->sl_cur->bkts = (EHBkt *)malloc((size_t)EH_SLAB_N * sizeof(EHBkt));
        t->sl_cur->used = 0;
    }
    EHBkt *b = &t->sl_cur->bkts[t->sl_cur->used++];
    b->cnt = 0;
    return b;
}

/* Allocate and initialise a fresh EHT with EH_INIT_GD initial depth. */
static EHT *eh_alloc(void) {
    EHT *t = (EHT *)calloc(1, sizeof(EHT));
    t->gd  = EH_INIT_GD;
    size_t dsz = (size_t)1 << EH_INIT_GD;
    t->dir = (EHBkt **)malloc(dsz * sizeof(EHBkt *));
    for (size_t i = 0; i < dsz; i++) {
        EHBkt *b = eh_bkt_alloc(t);
        b->ld    = EH_INIT_GD;
        t->dir[i] = b;
    }
    t->cnt = 0;
    return t;
}

/* Free all memory owned by t. */
static void eh_free(EHT *t) {
    EHSlab *sl = t->sl_head;
    while (sl) {
        EHSlab *nx = sl->next;
        free(sl->bkts);
        free(sl);
        sl = nx;
    }
    free(t->dir);
    free(t);
}

/* Reset t for reuse: free all slab bucket arrays and reinitialise directory.
   Freeing the bkts arrays (each ~3.2 MB) is the only reliable way to return
   pages to the OS on macOS: madvise(MADV_FREE) on malloc-backed memory is
   advisory and macOS ignores it in practice.  Large mallocs use mmap
   internally on macOS, so free() calls munmap and the OS actually reclaims
   the physical pages immediately, preventing 70+ GB swap accumulation.
   The EHSlab structs themselves (tiny: just used/next) are kept so the
   slab chain can be reused without rebuilding it from scratch.            */
static void eh_reset(EHT *t) {
    for (EHSlab *sl = t->sl_head; sl; sl = sl->next) {
        free(sl->bkts);   /* releases pages back to OS on macOS/Linux      */
        sl->bkts = NULL;
        sl->used = 0;
    }
    t->sl_cur = t->sl_head;
    t->cnt    = 0;

    /* Reinitialise directory at EH_INIT_GD (may shrink from a grown dir). */
    t->gd = EH_INIT_GD;
    size_t dsz = (size_t)1 << EH_INIT_GD;
    t->dir = (EHBkt **)realloc(t->dir, dsz * sizeof(EHBkt *));
    for (size_t i = 0; i < dsz; i++) {
        EHBkt *b = eh_bkt_alloc(t);
        b->ld    = EH_INIT_GD;
        t->dir[i] = b;
    }
}

/* Split the bucket currently at dir[dir_idx].
   Called when that bucket is full and a new insert is needed.

   Algorithm (standard extendible hashing):
   1. If local_depth == global_depth: double the directory.
   2. Increment local_depth of old bucket; allocate new sibling bucket.
   3. Redistribute entries: hash bit (64 - new_ld) selects old vs new.
   4. Update directory: all entries pointing to old where that bit is 1
      now point to new.                                                   */
static void eh_split(EHT *t, size_t dir_idx) {
    EHBkt *old = t->dir[dir_idx];   /* save before any directory changes */
    int    ld  = old->ld;

    /* 1. Double directory if needed. */
    if (ld == t->gd) {
        size_t old_dsz = (size_t)1 << t->gd;
        size_t new_dsz = old_dsz * 2;
        EHBkt **nd = (EHBkt **)malloc(new_dsz * sizeof(EHBkt *));
        for (size_t i = 0; i < old_dsz; i++) {
            nd[2*i]     = t->dir[i];
            nd[2*i + 1] = t->dir[i];
        }
        free(t->dir);
        t->dir = nd;
        t->gd++;
        dir_idx <<= 1;   /* both 2*old and 2*old+1 point to old; use either */
    }

    /* 2. Allocate sibling bucket and increment local depths. */
    int    new_ld = ld + 1;
    EHBkt *nw     = eh_bkt_alloc(t);
    nw->ld  = new_ld;
    old->ld = new_ld;

    /* 3. Redistribute entries based on bit (64 - new_ld) of their hash.
          Entries with that bit = 0 stay in old; bit = 1 go to nw.      */
    u64  ks[EH_BKT_CAP];
    u128 vs[EH_BKT_CAP];
    int  nkept = 0;
    for (int i = 0; i < old->cnt; i++) {
        u64 h = eh_hash(old->k[i]);
        if ((h >> (64 - new_ld)) & 1ULL) {
            nw->k[nw->cnt] = old->k[i];
            nw->v[nw->cnt] = old->v[i];
            nw->cnt++;
        } else {
            ks[nkept] = old->k[i];
            vs[nkept] = old->v[i];
            nkept++;
        }
    }
    old->cnt = nkept;
    memcpy(old->k, ks, (size_t)nkept * sizeof(u64));
    memcpy(old->v, vs, (size_t)nkept * sizeof(u128));

    /* 4. O(range/2) directory update — the critical fix.
          Previously we scanned all 2^gd directory entries looking for
          pointers to old (O(2^gd) per split → O(n_states * gd) total).
          With 141K splits at gd=18, that was 37 billion iterations.

          Correct approach: the directory entries pointing to old form a
          contiguous power-of-2-aligned range of size 2^(gd-ld).
          After the split we only need to redirect the upper half to nw.

          range = 2^(gd - ld)   entries formerly pointing to old
          base  = dir_idx & ~(range-1)  first such entry
          upper half [base + range/2, base + range) → redirect to nw

          Total amortised work: O(n_final_buckets) across all splits,
          because each directory entry is redirected at most once.        */
    size_t range = (size_t)1 << (t->gd - ld);
    size_t base  = dir_idx & ~(range - 1);
    for (size_t i = base + (range >> 1); i < base + range; i++)
        t->dir[i] = nw;
}

/* Insert key → accumulate val.  Splits automatically if bucket is full.
   Uses an iterative loop rather than recursion to avoid deep call stacks. */
static void eh_insert(EHT *t, u64 key, u128 val) {
    for (;;) {
        u64    h       = eh_hash(key);
        size_t dir_idx = (size_t)(h >> (64 - t->gd));
        EHBkt *bkt     = t->dir[dir_idx];

        /* Search bucket for existing key (3KB scan, L1-resident). */
        for (int i = 0; i < bkt->cnt; i++) {
            if (bkt->k[i] == key) { bkt->v[i] += val; return; }
        }

        /* Key not found: insert if there is room. */
        if (bkt->cnt < EH_BKT_CAP) {
            bkt->k[bkt->cnt] = key;
            bkt->v[bkt->cnt] = val;
            bkt->cnt++;
            t->cnt++;
            return;
        }

        /* Bucket full: split and retry. */
        eh_split(t, dir_idx);
    }
}

/* Return pointer to value for key, or NULL if not present. */
static u128 *eh_lookup(EHT *t, u64 key) {
    u64    h       = eh_hash(key);
    size_t dir_idx = (size_t)(h >> (64 - t->gd));
    EHBkt *bkt     = t->dir[dir_idx];
    for (int i = 0; i < bkt->cnt; i++)
        if (bkt->k[i] == key) return &bkt->v[i];
    return NULL;
}

/* ======================================================================
   Radix-sort insert buffer (RSortBuf)  — flat mmap edition
   -----------------------------------------------------------------------
   Groups outputs by top RSORT_BITS hash bits, then flushes slots 0…P-1
   in order so each EH directory range is warmed once per flush pass.

   MEMORY PROBLEM WITH malloc/free (root cause of 85 GB swap at n=61):
   The previous design called malloc(hint × 1024 × 24B) at the start of
   each sweep and free() at the end.  macOS's allocator retains freed pages
   in the heap rather than returning them to the OS, so over 33 steps with
   hint ≈ 50M/1024, this accumulated 33 × 2 sweeps × 1.2 GB = 79 GB of
   virtual memory, all of which macOS swapped to disk.

   FIX: allocate ONE flat buffer with mmap() at program startup.
   mmap(MAP_ANONYMOUS) pages are returned to the OS as soon as the mapping
   is released (munmap) or declared unneeded (madvise MADV_FREE / DONTNEED).
   We call madvise(MADV_FREE) after each flush pass so the OS can reclaim
   the physical pages immediately, even though the virtual mapping persists.

   Design:
   - Flat array of RSORT_FLAT_N WEntry objects, split into RSORT_SLOTS
     equal-sized partitions of RSORT_SLOT_CAP entries each.
   - Slot b owns entries [b*RSORT_SLOT_CAP … (b+1)*RSORT_SLOT_CAP).
   - Periodic flush triggers at RSORT_FLUSH_THRESH total entries and after
     every sweep, preserving hash-order locality.
   - Fixed per-slot capacity: if a slot overflows, flush immediately.
   - Total footprint: RSORT_FLAT_N × 24 B ≈ 480 MB (constant, no growth).
   ====================================================================== */
#ifndef RSORT_BITS
#define RSORT_BITS       10                     /* tuned at compile time     */
#endif
#define RSORT_SLOTS      (1 << RSORT_BITS)      /* 1024 partitions           */
#ifndef RSORT_THRESH
#define RSORT_THRESH     (1 << 19)              /* states threshold           */
#endif
#define RSORT_SLOT_CAP   (20 * 1024)            /* entries per slot: 20K     */
#define RSORT_FLAT_N     (RSORT_SLOTS * RSORT_SLOT_CAP) /* 20M entries total */
#define RSORT_FLUSH_THRESH (RSORT_FLAT_N / 2)   /* flush at ~10M total       */

typedef struct { u64 key; u128 val; } WEntry;

/* Global flat buffer — allocated once with mmap, reused every step.        */
static WEntry *g_rsort_flat = NULL;

static void rsort_global_init(void) {
    if (g_rsort_flat) return;
    size_t sz = (size_t)RSORT_FLAT_N * sizeof(WEntry);
    g_rsort_flat = (WEntry *)malloc(sz);
}

typedef struct {
    int    cnt[RSORT_SLOTS];   /* live count in each slot                    */
    size_t total;              /* total live entries                          */
    EHT   *t;
} RSortBuf;

static void rsort_init(RSortBuf *rb, EHT *t, size_t hint) {
    (void)hint;   /* ignored: slot capacity is now fixed at RSORT_SLOT_CAP  */
    rb->t     = t;
    rb->total = 0;
    memset(rb->cnt, 0, sizeof rb->cnt);
}

/* Flush all slots in hash order, reset counts, advise OS to free pages.    */
static void rsort_flush_reset(RSortBuf *rb) {
    for (int b = 0; b < RSORT_SLOTS; b++) {
        int n = rb->cnt[b];
        if (!n) continue;
        WEntry *base = g_rsort_flat + (size_t)b * RSORT_SLOT_CAP;
        for (int i = 0; i < n; i++)
            eh_insert(rb->t, base[i].key, base[i].val);
        rb->cnt[b] = 0;
    }
    rb->total = 0;
    /* Advise the OS that the flat buffer pages are no longer needed.
       Physical pages are reclaimed; virtual mapping stays.                  */
#if defined(MADV_FREE)
    madvise(g_rsort_flat,
            (size_t)RSORT_FLAT_N * sizeof(WEntry), MADV_FREE);
#elif defined(MADV_DONTNEED)
    madvise(g_rsort_flat,
            (size_t)RSORT_FLAT_N * sizeof(WEntry), MADV_DONTNEED);
#endif
}

static void rsort_flush(RSortBuf *rb) { rsort_flush_reset(rb); }

static inline void rsort_add(RSortBuf *rb, u64 key, u128 val) {
    int b = (int)(eh_hash(key) >> (64 - RSORT_BITS));
    int n = rb->cnt[b];
    if (n == RSORT_SLOT_CAP) {
        /* Slot full: flush everything in hash order then retry.             */
        rsort_flush_reset(rb);
        n = 0;
    }
    WEntry *base = g_rsort_flat + (size_t)b * RSORT_SLOT_CAP;
    base[n].key = key;
    base[n].val = val;
    rb->cnt[b]  = n + 1;
    rb->total++;
    if (rb->total >= RSORT_FLUSH_THRESH)
        rsort_flush_reset(rb);
}

/* Iterate all unique buckets via pool traversal (no directory dedup needed).
   Usage:
       EH_FOR(t, bkt) {
           for (int i = 0; i < bkt->cnt; i++) { ... bkt->k[i] ... bkt->v[i] ... }
       } EH_END
   Internal variables _esl_ and _ebi_ must not be shadowed in the body.    */
#define EH_FOR(t_, bkt_)                                                     \
    for (EHSlab *_esl_ = (t_)->sl_head; _esl_; _esl_ = _esl_->next)        \
        for (int _ebi_ = 0; _ebi_ < _esl_->used; _ebi_++) {                 \
            EHBkt *(bkt_) = &_esl_->bkts[_ebi_];

#define EH_END  }

/* ======================================================================
   apply_edge: apply one edge (v_idx, w_idx) to state key nk.
   Returns updated canonical key, or 0 if the edge is invalid.
   ====================================================================== */
static inline u64 apply_edge(u64 nk, int fs,
                              int v_idx, int w_idx, int *new_nc)
{
    int8_t sv = slot_get(nk, v_idx);
    int8_t sw = slot_get(nk, w_idx);

    if (sv == -1 || sw == -1) return 0;

    if (sv == 0 && sw == 0) {
        int8_t L = label_max(nk, fs) + 1;
        nk = slot_set(nk, v_idx, L);
        nk = slot_set(nk, w_idx, L);
        return canon(nk, fs);
    }
    if (sv == 0) {
        nk = slot_set(nk, w_idx, -1);
        nk = slot_set(nk, v_idx, sw);
        return canon(nk, fs);
    }
    if (sw == 0) {
        nk = slot_set(nk, v_idx, -1);
        nk = slot_set(nk, w_idx, sv);
        return canon(nk, fs);
    }
    if (sv == sw) return 0;

    int sv_c = label_count(nk, fs, sv);
    int sw_c = label_count(nk, fs, sw);
    nk = slot_set(nk, v_idx, -1);
    nk = slot_set(nk, w_idx, -1);
    nk = label_rename(nk, fs, sw, sv);

    if (sv_c == 1 && sw_c == 1) {
        if (nc_get(nk) >= 1) return 0;
        for (int k = 0; k < fs; k++)
            if (slot_get(nk, k) != -1) return 0;
        *new_nc = 1;
        return canon(nc_set_1(nk), fs);
    }
    return canon(nk, fs);
}

/* ======================================================================
   apply_elim_seq: apply n_elim eliminations (descending index order).
   Returns updated key, or 0 (invalid / final-step count accumulated).
   ====================================================================== */
static inline u64 apply_elim_seq(u64 nk, int cur_fs,
                                  const int *elim_idxs_desc, int n_elim,
                                  int step, int n, u128 cnt, u128 *total)
{
    for (int e = 0; e < n_elim; e++) {
        int    u_idx = elim_idxs_desc[e];
        int8_t su    = slot_get(nk, u_idx);

        if (su == 0) return 0;

        u64 rk = eliminate_slot(nk, u_idx, cur_fs);
        cur_fs--;

        if (su == -1) {
            nk = canon(rk, cur_fs);
        } else {
            int partner = 0;
            for (int k = 0; k < cur_fs; k++)
                if (slot_get(rk, k) == su) { partner = 1; break; }

            if (partner) {
                nk = canon(rk, cur_fs);
            } else {
                if (nc_get(nk) >= 1) return 0;
                for (int k = 0; k < cur_fs; k++)
                    if (slot_get(rk, k) != -1) return 0;

                nk = nc_set_1(canon(rk, cur_fs));
                if (cur_fs == 0) {
                    if (step == n - 1) *total += cnt;
                    return 0;
                }
            }
        }
    }

    if (cur_fs == 0) {
        if (step == n - 1 && nc_get(nk) == 1) *total += cnt;
        return 0;
    }
    return nk;
}

/* ======================================================================
   fused_sweep: single source-table scan applying 2^n_back edge subsets
   AND n_elim eliminations to each state.  Writes results into nxt.

   nxt is reset at the start of this function.
   No write-combining buffer is needed: EH already groups entries by hash
   prefix, so cache locality comes for free from the bucket structure.
   ====================================================================== */
#define FUSED_MIN_STATES 200000

static void fused_sweep(EHT *curr, EHT *nxt,
                        int fs, int v_idx,
                        const int *widxs, int n_back,
                        const int *elim_idxs_desc, int n_elim,
                        int step, int n, u128 *total)
{
    int n_subsets = 1 << n_back;

    eh_reset(nxt);

    int use_rsort = (curr->cnt >= RSORT_THRESH);
    RSortBuf rb = {0};
    if (use_rsort) rsort_init(&rb,   nxt, 0);

    EH_FOR(curr, bkt) {
        for (int i = 0; i < bkt->cnt; i++) {
            u64  base = bkt->k[i];
            u128 cnt  = bkt->v[i];

            for (int S = 0; S < n_subsets; S++) {
                u64 nk    = base;
                int valid = 1;

                for (int j = 0; j < n_back && valid; j++) {
                    if (!(S & (1 << j))) continue;
                    int nc_inc = 0;
                    u64 nk2 = apply_edge(nk, fs, v_idx, widxs[j], &nc_inc);
                    if (!nk2) { valid = 0; break; }
                    nk = nk2;
                }
                if (!valid) continue;

                if (n_elim > 0) {
                    nk = apply_elim_seq(nk, fs, elim_idxs_desc, n_elim,
                                        step, n, cnt, total);
                    if (!nk) continue;
                }

                if (use_rsort) rsort_add(&rb, nk, cnt);
                else           eh_insert(nxt, nk, cnt);
            }
        }
    } EH_END

    if (use_rsort) rsort_flush(&rb);
}

/* ======================================================================
   Checkpointing
   -----------------------------------------------------------------------
   Binary format (little-endian, same as LP version v2):
     4B  magic   0xC8EC7A1E
     4B  version 2
     4B  n
     4B  step    (last completed step)
     4B  fs      (frontier size after step)
     8B  ordering hash (FNV-1a of order[])
     8B  total_lo
     8B  total_hi
     fs×4  frontier[]
     8B  cnt     (number of entries)
     cnt × (8+8+8)  key, val_lo, val_hi
   ====================================================================== */
#define CKPT_MAGIC   0xC8EC7A1Eu
#define CKPT_VERSION 2u

static uint64_t order_hash(int n, const int *order)
{
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)order[i];
        h ^= (v & 0xff);         h *= 1099511628211ULL;
        h ^= ((v >>  8) & 0xff); h *= 1099511628211ULL;
        h ^= ((v >> 16) & 0xff); h *= 1099511628211ULL;
        h ^= ((v >> 24) & 0xff); h *= 1099511628211ULL;
    }
    return h;
}

static int ckpt_save(const char *path, int n, const int *order,
                     int step, int fs,
                     const int *frontier, u128 total, EHT *curr)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    uint32_t magic   = CKPT_MAGIC;
    uint32_t version = CKPT_VERSION;
    uint32_t un      = (uint32_t)n;
    uint32_t ustep   = (uint32_t)step;
    uint32_t ufs     = (uint32_t)fs;
    uint64_t ord_h   = order_hash(n, order);
    uint64_t tot_lo  = (uint64_t)total;
    uint64_t tot_hi  = (uint64_t)(total >> 64);
    uint64_t cnt     = (uint64_t)curr->cnt;

    if (fwrite(&magic,   4, 1, f) != 1) goto fail;
    if (fwrite(&version, 4, 1, f) != 1) goto fail;
    if (fwrite(&un,      4, 1, f) != 1) goto fail;
    if (fwrite(&ustep,   4, 1, f) != 1) goto fail;
    if (fwrite(&ufs,     4, 1, f) != 1) goto fail;
    if (fwrite(&ord_h,   8, 1, f) != 1) goto fail;
    if (fwrite(&tot_lo,  8, 1, f) != 1) goto fail;
    if (fwrite(&tot_hi,  8, 1, f) != 1) goto fail;
    if (fs > 0 && fwrite(frontier, 4*(size_t)fs, 1, f) != 1) goto fail;
    if (fwrite(&cnt,     8, 1, f) != 1) goto fail;

    /* Write occupied entries via pool traversal. */
    EH_FOR(curr, bkt) {
        for (int i = 0; i < bkt->cnt; i++) {
            uint64_t val_lo = (uint64_t) bkt->v[i];
            uint64_t val_hi = (uint64_t)(bkt->v[i] >> 64);
            if (fwrite(&bkt->k[i], 8, 1, f) != 1) goto fail;
            if (fwrite(&val_lo,    8, 1, f) != 1) goto fail;
            if (fwrite(&val_hi,    8, 1, f) != 1) goto fail;
        }
    } EH_END

    fclose(f);
    return 1;
fail:
    fclose(f);
    return 0;
}

static int ckpt_load(const char *path, int n_expected,
                     const int *order,
                     int *step_out, int *fs_out, int *frontier_out,
                     u128 *total_out, EHT *curr)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t magic, version, un, ustep, ufs;
    uint64_t ord_h_saved, tot_lo, tot_hi, cnt;

    if (fread(&magic,   4, 1, f) != 1 || magic   != CKPT_MAGIC)   goto fail;
    if (fread(&version, 4, 1, f) != 1 || version != CKPT_VERSION)  goto fail;
    if (fread(&un,      4, 1, f) != 1 || (int)un != n_expected)    goto fail;
    if (fread(&ustep,   4, 1, f) != 1) goto fail;
    if (fread(&ufs,     4, 1, f) != 1) goto fail;
    if (fread(&ord_h_saved, 8, 1, f) != 1) goto fail;

    if (ord_h_saved != order_hash(n_expected, order)) {
        fprintf(stderr,
            "# Checkpoint ordering mismatch — discarding stale checkpoint.\n");
        goto fail;
    }

    if (fread(&tot_lo, 8, 1, f) != 1) goto fail;
    if (fread(&tot_hi, 8, 1, f) != 1) goto fail;

    *step_out  = (int)ustep;
    *fs_out    = (int)ufs;
    *total_out = ((u128)tot_hi << 64) | (u128)tot_lo;

    if ((int)ufs > 0) {
        if (fread(frontier_out, 4*(size_t)ufs, 1, f) != 1) goto fail;
    }
    if (fread(&cnt, 8, 1, f) != 1) goto fail;

    eh_reset(curr);
    for (uint64_t e = 0; e < cnt; e++) {
        uint64_t key, val_lo, val_hi;
        if (fread(&key,    8, 1, f) != 1) goto fail;
        if (fread(&val_lo, 8, 1, f) != 1) goto fail;
        if (fread(&val_hi, 8, 1, f) != 1) goto fail;
        u128 val = ((u128)val_hi << 64) | (u128)val_lo;
        eh_insert(curr, key, val);
    }
    fclose(f);
    return 1;
fail:
    fclose(f);
    return 0;
}

/* ======================================================================
   Main frontier DP
   ====================================================================== */
void count_ham_paths_c(
    int        n,
    const int *order,
    const int *pos,
    const int *last_s,
    const int *adj_off,
    const int *adj_dat,
    uint64_t  *res_lo,
    uint64_t  *res_hi,
    int        verbose,
    const char *checkpoint_path,
    double     checkpoint_secs
) {
    rsort_global_init();   /* ensure flat rsort buffer is mmap'd             */
    int  frontier[MAX_FS_FAST + 2];
    int *fidx = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int  fs   = 0;
    for (int i = 0; i <= n; i++) fidx[i] = -1;

    EHT *curr = eh_alloc();
    EHT *nxt  = eh_alloc();

    u128 total = (u128)0;
    int  start_step = 0;

    /* Try to resume from checkpoint. */
    if (checkpoint_path && checkpoint_path[0]) {
        int ckpt_step = -1, ckpt_fs = 0;
        if (ckpt_load(checkpoint_path, n, order, &ckpt_step, &ckpt_fs,
                      frontier, &total, curr)) {
            start_step = ckpt_step + 1;
            fs = ckpt_fs;
            for (int i = 0; i < fs; i++) fidx[frontier[i]] = i;
            if (verbose)
                fprintf(stderr, "# Resumed from checkpoint at step %d (%zu states)\n",
                        ckpt_step, curr->cnt);
        }
    }

    if (start_step == 0)
        eh_insert(curr, KEY_MARKER, (u128)1);

    double t_start = now_ms(), t_prev = t_start, t_last_ckpt = t_start;

    if (verbose)
        fprintf(stderr,
            "step  vertex  fs  n_back  states_in  states_out"
            "  step_ms  cumul_ms\n");

    for (int step = start_step; step < n; step++) {
        int v = order[step];

        if (fs >= MAX_FS_FAST) {
            *res_lo = *res_hi = UINT64_MAX;
            goto cleanup;
        }

        /* ---- A. Introduce v ------------------------------------------ */
        {
            int use_rsort_a = (curr->cnt >= RSORT_THRESH);
            RSortBuf rb_a = {0};
            eh_reset(nxt);
            if (use_rsort_a) rsort_init(&rb_a, nxt, 0);
            EH_FOR(curr, bkt) {
                for (int i = 0; i < bkt->cnt; i++) {
                    u64 nk = introduce(bkt->k[i], fs);
                    if (use_rsort_a) rsort_add(&rb_a, nk, bkt->v[i]);
                    else             eh_insert(nxt, nk, bkt->v[i]);
                }
            } EH_END
            if (use_rsort_a) rsort_flush(&rb_a);
        }
        { EHT *tmp = curr; curr = nxt; nxt = tmp; }
        frontier[fs] = v;
        fidx[v]      = fs;
        fs++;

        /* ---- B + C. Edge decisions and elimination ------------------- */
        int    v_idx     = fs - 1;
        size_t states_in = curr->cnt;
        int    n_back    = 0;
        int    widxs[8];

        for (int ai = adj_off[v]; ai < adj_off[v + 1]; ai++) {
            int w = adj_dat[ai];
            if (fidx[w] >= 0 && pos[w] < pos[v] && n_back < 8)
                widxs[n_back++] = fidx[w];
        }

        int elim_idxs_asc[MAX_FS_FAST + 2], n_elim = 0;
        for (int ei = 0; ei < fs; ei++)
            if (last_s[frontier[ei]] <= step)
                elim_idxs_asc[n_elim++] = ei;
        int elim_idxs_desc[MAX_FS_FAST + 2];
        for (int e = 0; e < n_elim; e++)
            elim_idxs_desc[e] = elim_idxs_asc[n_elim - 1 - e];

        /* Fused path: edges + eliminations in one source-table scan.
           Non-fused path: sequential per-edge sweeps then elimination.    */
        int use_fused = (n_elim >= 1) &&
                        (curr->cnt >= FUSED_MIN_STATES || n_back >= 2);

        if (use_fused) {
            fused_sweep(curr, nxt, fs, v_idx,
                        widxs, n_back,
                        elim_idxs_desc, n_elim,
                        step, n, &total);
            { EHT *tmp = curr; curr = nxt; nxt = tmp; }

        } else {
            /* B: sequential per-back-edge sweeps. */
            for (int j = 0; j < n_back; j++) {
                int w_idx = widxs[j];
                int use_rsort_b = (curr->cnt >= RSORT_THRESH);
                RSortBuf rb_b = {0};
                eh_reset(nxt);
                if (use_rsort_b) rsort_init(&rb_b, nxt, 0);
                EH_FOR(curr, bkt) {
                    for (int i = 0; i < bkt->cnt; i++) {
                        u64    key = bkt->k[i];
                        u128   cnt = bkt->v[i];
                        int8_t sv  = slot_get(key, v_idx);
                        int8_t sw  = slot_get(key, w_idx);
#define BINSERT(k_) do { if(use_rsort_b) rsort_add(&rb_b,(k_),cnt); \
                         else eh_insert(nxt,(k_),cnt); } while(0)
                        BINSERT(key);     /* always copy base */
                        if (sv == -1 || sw == -1) { (void)0; } else {
                        u64 nk = key;
                        if (sv == 0 && sw == 0) {
                            int8_t L = label_max(key, fs) + 1;
                            nk = slot_set(nk, v_idx, L);
                            nk = slot_set(nk, w_idx, L);
                            BINSERT(canon(nk, fs));
                        } else if (sv == 0) {
                            nk = slot_set(nk, w_idx, -1);
                            nk = slot_set(nk, v_idx, sw);
                            BINSERT(canon(nk, fs));
                        } else if (sw == 0) {
                            nk = slot_set(nk, v_idx, -1);
                            nk = slot_set(nk, w_idx, sv);
                            BINSERT(canon(nk, fs));
                        } else if (sv != sw) {
                            int sv_c = label_count(key, fs, sv);
                            int sw_c = label_count(key, fs, sw);
                            nk = slot_set(nk, v_idx, -1);
                            nk = slot_set(nk, w_idx, -1);
                            nk = label_rename(nk, fs, sw, sv);
                            if (sv_c == 1 && sw_c == 1) {
                                if (nc_get(nk) >= 1) goto bskip;
                                int ok = 1;
                                for (int k2 = 0; k2 < fs; k2++)
                                    if (slot_get(nk, k2) != -1) { ok = 0; break; }
                                if (!ok) goto bskip;
                                BINSERT(canon(nc_set_1(nk), fs));
                            } else {
                                BINSERT(canon(nk, fs));
                            }
                        }
                        }
                        bskip:;
#undef BINSERT
                    }
                } EH_END
                if (use_rsort_b) rsort_flush(&rb_b);
                { EHT *tmp = curr; curr = nxt; nxt = tmp; }
            }

            /* C: eliminate vertices one at a time (ascending order). */
            for (int e = 0; e < n_elim; e++) {
                int u_idx = elim_idxs_asc[e];
                int use_rsort_c = (curr->cnt >= RSORT_THRESH);
                RSortBuf rb_c = {0};
                eh_reset(nxt);
                if (use_rsort_c) rsort_init(&rb_c, nxt, 0);
#define CINSERT(k_,v_) do { if(use_rsort_c) rsort_add(&rb_c,(k_),(v_)); \
                            else eh_insert(nxt,(k_),(v_)); } while(0)
                EH_FOR(curr, bkt) {
                    for (int i = 0; i < bkt->cnt; i++) {
                        u64    key = bkt->k[i];
                        u128   cnt = bkt->v[i];
                        int8_t su  = slot_get(key, u_idx);
                        int    nc  = nc_get(key);
                        if (su == 0) continue;
                        u64 rk = eliminate_slot(key, u_idx, fs);
                        if (su == -1) {
                            CINSERT(canon(rk, fs - 1), cnt);
                        } else {
                            int partner = 0;
                            for (int k2 = 0; k2 < fs - 1; k2++)
                                if (slot_get(rk, k2) == su) { partner = 1; break; }
                            if (partner) {
                                CINSERT(canon(rk, fs - 1), cnt);
                            } else {
                                if (nc + 1 > 1) continue;
                                int ok = 1;
                                for (int k2 = 0; k2 < fs - 1; k2++)
                                    if (slot_get(rk, k2) != -1) { ok = 0; break; }
                                if (!ok) continue;
                                if (step == n - 1) total += cnt;
                            }
                        }
                    }
                } EH_END
#undef CINSERT
                if (use_rsort_c) rsort_flush(&rb_c);
                { EHT *tmp = curr; curr = nxt; nxt = tmp; }
                fs--;
                int u = frontier[u_idx];
                for (int k = u_idx; k < fs; k++) {
                    frontier[k] = frontier[k + 1];
                    fidx[frontier[k]] = k;
                }
                fidx[u] = -1;
                for (int e2 = e + 1; e2 < n_elim; e2++)
                    if (elim_idxs_asc[e2] > u_idx)
                        elim_idxs_asc[e2]--;
            }
        }

        /* Update frontier for the fused path (non-fused does it above). */
        if (use_fused) {
            for (int e = 0; e < n_elim; e++) {
                int u_idx = elim_idxs_desc[e];
                int u     = frontier[u_idx];
                for (int k = u_idx; k < fs - 1; k++) {
                    frontier[k] = frontier[k + 1];
                    fidx[frontier[k]] = k;
                }
                fidx[u] = -1;
                fs--;
            }
        }

        /* Final-step lookup: count Hamiltonian path completions. */
        if (step == n - 1) {
            u64   target = KEY_MARKER | KEY_NC_BIT;
            u128 *vp     = eh_lookup(curr, target);
            if (vp) total += *vp;
        }

        if (verbose) {
            double t_now = now_ms();
            fprintf(stderr,
                "%4d  %6d  %2d  %6d  %9zu  %10zu  %8.1f  %8.1f\n",
                step, v, fs, n_back,
                states_in, curr->cnt,
                t_now - t_prev, t_now - t_start);
            t_prev = t_now;
        }

        if (checkpoint_path && checkpoint_path[0] && checkpoint_secs > 0) {
            double t_now2 = now_ms();
            if ((t_now2 - t_last_ckpt) / 1000.0 >= checkpoint_secs) {
                if (ckpt_save(checkpoint_path, n, order, step, fs,
                              frontier, total, curr)) {
                    if (verbose)
                        fprintf(stderr, "# Checkpoint saved at step %d (%zu states)\n",
                                step, curr->cnt);
                    t_last_ckpt = t_now2;
                }
            }
        }
    }

cleanup:
    eh_free(curr);
    eh_free(nxt);
    free(fidx);

    if (*res_lo != UINT64_MAX || *res_hi != UINT64_MAX) {
        *res_lo = (uint64_t) total;
        *res_hi = (uint64_t)(total >> 64);
    }
}
"""

# ---------------------------------------------------------------------------
# Cache geometry detection and compile-time constant derivation
# ---------------------------------------------------------------------------

def _detect_cache_sizes():
    """Return (l1d_bytes, l2_bytes, l3_bytes, cacheline_bytes).

    Tries:
      macOS  — sysctl hw.l1dcachesize / hw.l2cachesize / hw.l3cachesize
               Apple Silicon exposes hw.l3cachesize directly; older Macs may
               not have L3, so we fall back to hw.cachesize (total).
      Linux  — /sys/devices/system/cpu/cpu0/cache/index*/  (most reliable)
               then /proc/cpuinfo 'cache size' line as a fallback.
    Falls back to conservative x86 defaults on failure.
    """
    import platform, re
    defaults = (32 << 10, 512 << 10, 8 << 20, 64)   # 32KB / 512KB / 8MB / 64B

    def sysctl_int(key):
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        try: return int(r.stdout.strip()) if r.returncode == 0 else None
        except ValueError: return None

    sys_ = platform.system()
    if sys_ == "Darwin":
        l1  = sysctl_int("hw.l1dcachesize")
        l2  = sysctl_int("hw.l2cachesize")
        l3  = sysctl_int("hw.l3cachesize") or sysctl_int("hw.cachesize")
        cl  = sysctl_int("hw.cachelinesize")
        return (l1 or defaults[0], l2 or defaults[1],
                l3 or defaults[2], cl or defaults[3])

    elif sys_ == "Linux":
        l1d = l2 = l3 = None
        cl = 64
        for idx in range(8):
            base = f"/sys/devices/system/cpu/cpu0/cache/index{idx}"
            if not os.path.exists(base):
                break
            try:
                level  = int(open(f"{base}/level").read())
                ctype  = open(f"{base}/type").read().strip()
                s_str  = open(f"{base}/size").read().strip()
                cl     = int(open(f"{base}/coherency_line_size").read())
                mult   = (1 << 20) if s_str.endswith("M") else (1 << 10)
                size   = int(s_str.rstrip("KMG")) * mult
                if level == 1 and "Data" in ctype: l1d = size
                elif level == 2:                   l2  = size
                elif level == 3:                   l3  = size
            except Exception:
                continue
        if l3 is None:                              # /proc/cpuinfo fallback
            try:
                for line in open("/proc/cpuinfo"):
                    m = re.search(r"cache size\s*:\s*(\d+)\s*KB", line)
                    if m: l3 = int(m.group(1)) << 10; break
            except Exception:
                pass
        return (l1d or defaults[0], l2 or defaults[1],
                l3 or defaults[2], cl)

    return defaults


def _derive_build_constants(l1d: int, l2: int, l3: int, cl: int) -> dict:
    """Derive EH / rsort compile-time constants from cache geometry.

    EH_BKT_CAP:
        A bucket must fit comfortably in L1 so the sequential key scan is
        cache-resident.  32 entries × 24 B = 768 B (12 cache lines at 64 B)
        is well within any L1 ≥ 16 KB.  We use 64 only if L1 ≥ 64 KB
        (uncommon, but some server parts have this).

    RSORT_THRESH:
        Activate the radix-sort buffer when the EH bucket pool would exceed
        L3 — at that point random eh_insert calls become DRAM-bound.
        pool_bytes ≈ states × (bkt_cap × 24 + 8) / (bkt_cap × 0.75)
        Solve for states: thresh = L3 / bytes_per_state, rounded to pow2.

    RSORT_BITS (→ RSORT_SLOTS = 2^RSORT_BITS):
        Each rsort flush processes all entries for a contiguous range of
        2^(gd - RSORT_BITS) EH buckets.  We want that flush region to fit
        in L2 with ~16 regions simultaneously warm.
        Sizing at the 30 M-state design point (covers n=58 peak; conservative
        for n=61):
            flush_bytes = (30M / fill / 2^bits) × bkt_bytes
            target      = L2 / 16
        Clamped to [8, 12] (256 … 4096 slots) to avoid excessive malloc.
    """
    import math
    bkt_cap = 64 if l1d >= (64 << 10) else 32
    bkt_bytes = bkt_cap * 24 + 8
    bps = bkt_bytes / (bkt_cap * 0.75)           # bytes per state in pool

    thresh_raw = int(l3 / bps)
    thresh = 1 << max(0, thresh_raw.bit_length() - 1)   # round down to pow2

    design_peak  = 30_000_000
    total_pool   = design_peak / (bkt_cap * 0.75) * bkt_bytes
    flush_target = max(l2 // 16, 64 << 10)        # L2/16, floor 64 KB
    bits_raw     = math.ceil(math.log2(total_pool / flush_target))
    bits         = max(8, min(12, bits_raw))

    return {"EH_BKT_CAP": bkt_cap, "RSORT_THRESH": thresh, "RSORT_BITS": bits}


# ---------------------------------------------------------------------------
_LIB_CACHE: dict = {}

def _get_lib():
    # Detect cache sizes once; derive -D flags; fold both into the cache key
    # so that recompilation happens automatically on a different machine.
    l1d, l2, l3, cl = _detect_cache_sizes()
    consts = _derive_build_constants(l1d, l2, l3, cl)

    src_hash = hashlib.md5(C_SOURCE.encode()).hexdigest()[:12]
    const_sig = "_".join(f"{k}{v}" for k, v in sorted(consts.items()))
    cache_key = f"{src_hash}_{const_sig}"
    if cache_key in _LIB_CACHE:
        return _LIB_CACHE[cache_key]

    build_dir = os.path.join(tempfile.gettempdir(), f"ham_dp_c_{cache_key}")
    os.makedirs(build_dir, exist_ok=True)
    c_path  = os.path.join(build_dir, "ham_dp.c")
    so_path = os.path.join(build_dir, "ham_dp.so")

    if not os.path.exists(so_path):
        with open(c_path, "w") as f:
            f.write(C_SOURCE)
        d_flags = [f"-D{k}={v}" for k, v in consts.items()]
        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-std=c11"]
            + d_flags
            + ["-o", so_path, c_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gcc failed:\n{result.stderr.decode()}")

    ffi = cffi.FFI()
    ffi.cdef("""
        void count_ham_paths_c(
            int n, const int *order, const int *pos, const int *last_s,
            const int *adj_off, const int *adj_dat,
            uint64_t *res_lo, uint64_t *res_hi,
            int verbose,
            const char *checkpoint_path, double checkpoint_secs);
    """)
    lib = ffi.dlopen(so_path)
    _LIB_CACHE[cache_key] = (lib, ffi)
    return lib, ffi


def count_hamiltonian_paths_c(n: int, order: list, adj: dict,
                               verbose: bool = False,
                               mem_reserve_gb: float = 2.0,  # kept for API compat; EH ignores it
                               load_factor: int = 75,        # kept for API compat; EH ignores it
                               checkpoint_path: str = "",
                               checkpoint_secs: float = 300.0) -> int:
    """Count undirected Hamiltonian paths in G_n via the C frontier DP.

    Uses extendible hashing for the state table — memory scales with actual
    entry count, no pre-allocation mismatch, no 3-table resize spikes.

    Parameters
    ----------
    n                : graph size
    order            : vertex ordering (1-indexed), length n.
    adj              : adjacency dict {v: iterable_of_neighbours} (1-indexed).
    verbose          : if True, print per-step profiling to stderr.
    mem_reserve_gb   : ignored (EH adapts to actual entry count automatically).
    load_factor      : ignored (EH has no fixed load factor).
    checkpoint_path  : path for checkpoint file ('' = disabled).
    checkpoint_secs  : checkpoint interval in seconds (0 = disable).
    """
    import sys
    if verbose and (load_factor != 75 or mem_reserve_gb != 2.0):
        print("# Note: EH backend ignores --load-factor and --mem-reserve "
              "(memory is proportional to actual entry count)", file=sys.stderr)

    lib, ffi = _get_lib()

    pos = [0] * (n + 1)
    for i, v in enumerate(order): pos[v] = i

    last_s = [0] * (n + 1)
    for v in range(1, n + 1):
        nbr_pos = [pos[w] for w in adj[v]]
        last_s[v] = max(nbr_pos) if nbr_pos else pos[v]

    adj_off_list = [0] * (n + 2)
    for v in range(1, n + 1):
        adj_off_list[v + 1] = adj_off_list[v] + len(adj[v])
    adj_dat_list: list = []
    for v in range(1, n + 1):
        adj_dat_list.extend(sorted(adj[v]))

    c_order   = ffi.new("int[]", list(order))
    c_pos     = ffi.new("int[]", pos)
    c_last_s  = ffi.new("int[]", last_s)
    c_adj_off = ffi.new("int[]", adj_off_list)
    c_adj_dat = ffi.new("int[]", adj_dat_list if adj_dat_list else [0])
    c_res_lo  = ffi.new("uint64_t*")
    c_res_hi  = ffi.new("uint64_t*")
    c_ckpt    = ffi.new("char[]", checkpoint_path.encode() if checkpoint_path else b"")

    lib.count_ham_paths_c(
        n, c_order, c_pos, c_last_s, c_adj_off, c_adj_dat,
        c_res_lo, c_res_hi, int(verbose),
        c_ckpt, ffi.cast("double", checkpoint_secs),
    )
    lo, hi = int(c_res_lo[0]), int(c_res_hi[0])
    if lo == hi == 0xFFFFFFFFFFFFFFFF:
        raise RuntimeError(
            "Frontier size exceeded MAX_FS_FAST=15 (pathwidth > 15). "
            "The packed uint64 state encoding is limited to 15 frontier slots."
        )
    return (hi << 64) | lo

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, time
    start_n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    end_n   = int(sys.argv[2]) if len(sys.argv) > 2 else start_n
    try:
        from ham_ordering import build_graph, best_bfs_order, frontier_stats
    except ImportError:
        from math import isqrt
        def build_graph(n):
            adj = {v: set() for v in range(1, n+1)}
            for i in range(1, n+1):
                for j in range(i+1, n+1):
                    s=i+j; r=isqrt(s)
                    if r*r==s: adj[i].add(j); adj[j].add(i)
            return adj
        def best_bfs_order(adj, n):
            import networkx as nx
            G = nx.Graph([(u,v) for u,ns in adj.items() for v in ns])
            nodes = list(range(1,n+1))
            best=(999,999,None)
            for s in nodes:
                o=list(nx.bfs_tree(G,s).nodes())+sorted(set(nodes)-set(nx.bfs_tree(G,s).nodes()))
                pos={v:i for i,v in enumerate(o)}
                fw=[sum(1 for u in o[:i+1] if any(pos[nb]>i for nb in adj[u])) for i in range(n)]
                if (max(fw),sum(fw))<best[:2]: best=(max(fw),sum(fw),o)
            return best[2]
        def frontier_stats(adj, order):
            pos={v:i for i,v in enumerate(order)}
            fw=[sum(1 for u in order[:i+1] if any(pos[nb]>i for nb in adj[u])) for i in range(len(order))]
            return max(fw),sum(fw)
    print("# Compiling …", flush=True); _get_lib(); print("# Ready.", flush=True)
    for n in range(start_n, end_n + 1):
        adj=build_graph(n); order=best_bfs_order(adj,n); mx,_=frontier_stats(adj,order)
        t0=time.time(); count=count_hamiltonian_paths_c(n,order,adj,verbose=True)
        print(f"n={n:3d}  max_fw={mx:2d}  ham_paths={count}  ({time.time()-t0:.3f}s)", flush=True)
