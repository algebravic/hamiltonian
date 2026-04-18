#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#  define _DARWIN_C_SOURCE   /* expose MAP_ANON and other BSD extensions */
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

/* ── macOS 26 xzone allocator workaround ─────────────────────────────────
   macOS 26 (Tahoe) introduced "xzone" malloc, which uses mach_vm_reclaim
   for deferred page reclamation.  Any malloc/free of a large block (≥ 1 MB)
   can trigger a kernel assertion failure:
     "BUG IN LIBMALLOC: malloc assertion err == VM_RECLAIM_SUCCESS failed"

   Fix: bypass malloc/free for large allocations on macOS by using
   mmap/munmap directly.  We store the mmap'd size in a 16-byte header
   immediately before the returned pointer so sm_free can find it.

   To distinguish mmap'd from malloc'd pointers in sm_free we use
   malloc_size(p): it returns the usable malloc size for malloc'd pointers
   and 0 for anything else (including mmap).  This is robust against
   the false-positive risk of a magic-sentinel approach where malloc's own
   metadata bytes could accidentally equal the sentinel value.

   sm_alloc(n)          — allocate n bytes
   sm_free(p)           — free any sm_alloc'd pointer
   sm_calloc(n)         — allocate n zero-filled bytes
   sm_realloc(p, new_n) — resize                                          */

#ifdef __APPLE__
#include <malloc/malloc.h>   /* malloc_size() — macOS only */
#endif

#define SM_LARGE_THRESHOLD ((size_t)(1 << 20))   /* 1 MB */
#define SM_HEADER_SIZE     16    /* one size_t (stored size) + 8 bytes pad  */

#ifdef __APPLE__

static inline void *sm_alloc(size_t n) {
    if (n == 0) return NULL;
    if (n < SM_LARGE_THRESHOLD) return malloc(n);
    size_t total = n + SM_HEADER_SIZE;
    void *raw = mmap(NULL, total, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    ((size_t *)raw)[0] = total;   /* store total mmap'd bytes in header */
    return (char *)raw + SM_HEADER_SIZE;
}

static inline void sm_free(void *p) {
    if (!p) return;
    /* malloc_size returns 0 for non-malloc pointers (mmap, stack, etc.).
       This is the canonical macOS way to distinguish malloc from mmap.   */
    if (malloc_size(p) > 0) {
        free(p);
    } else {
        size_t *hdr = (size_t *)((char *)p - SM_HEADER_SIZE);
        munmap(hdr, hdr[0]);   /* hdr[0] = total bytes including header  */
    }
}

static inline void *sm_calloc(size_t n) {
    if (n == 0) return NULL;
    if (n < SM_LARGE_THRESHOLD) return calloc(1, n);
    return sm_alloc(n);   /* mmap is always zero-initialised */
}

static inline void *sm_realloc(void *p, size_t new_n) {
    if (!p) return sm_alloc(new_n);
    if (new_n == 0) { sm_free(p); return NULL; }
    if (malloc_size(p) > 0) {
        /* malloc'd block */
        if (new_n < SM_LARGE_THRESHOLD) return realloc(p, new_n);
        /* Growing past threshold: migrate to mmap */
        size_t old_n = malloc_size(p);
        void *np = sm_alloc(new_n);
        if (!np) return NULL;
        memcpy(np, p, old_n < new_n ? old_n : new_n);
        free(p);
        return np;
    } else {
        /* mmap'd block */
        size_t *hdr      = (size_t *)((char *)p - SM_HEADER_SIZE);
        size_t old_total = hdr[0];
        size_t old_n     = old_total - SM_HEADER_SIZE;
        if (new_n <= old_n) {
            /* Shrink: update stored size, keep mapping */
            hdr[0] = new_n + SM_HEADER_SIZE;
            return p;
        }
        /* Grow: allocate new, copy, unmap old */
        void *np = sm_alloc(new_n);
        if (!np) return NULL;
        memcpy(np, p, old_n);
        munmap(hdr, old_total);
        return np;
    }
}

#else   /* Linux: plain wrappers, zero overhead */

#define SM_HEADER_SIZE 0
static inline void *sm_alloc(size_t n)          { return n ? malloc(n)   : NULL; }
static inline void  sm_free(void *p)             { free(p); }
static inline void *sm_calloc(size_t n)          { return n ? calloc(1,n): NULL; }
static inline void *sm_realloc(void *p, size_t n){ return realloc(p, n); }

#endif  /* __APPLE__ */

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

typedef unsigned __int128 u128;
typedef uint64_t          u64;

/* ── Runtime configuration: replaces compile-time -D SM_ constants ───
   Set once via sm_configure() (called from Python after dlopen).      */
typedef struct {
    int    nthreads; size_t worker_cap; int rbuf_size; int bb_buf;
    size_t ext_stream_buf; size_t slc_bytes;
    size_t par_merge_thresh; int steal_factor; size_t rsort_thresh;
} SMConfig;
static SMConfig sm_cfg = {
    .nthreads=6,.worker_cap=16*1024*1024,.rbuf_size=32,.bb_buf=32,
    .ext_stream_buf=1024,.slc_bytes=12*1024*1024,
    .par_merge_thresh=50000000,.steal_factor=32,.rsort_thresh=1<<19
};
void sm_configure(int nthreads,size_t worker_cap,int rbuf_size,
                  int bb_buf,size_t ext_stream_buf,size_t slc_bytes,
                  size_t par_merge_thresh,int steal_factor,
                  size_t rsort_thresh) {
    sm_cfg.nthreads=nthreads; sm_cfg.worker_cap=worker_cap;
    sm_cfg.rbuf_size=rbuf_size; sm_cfg.bb_buf=bb_buf;
    sm_cfg.ext_stream_buf=ext_stream_buf; sm_cfg.slc_bytes=slc_bytes;
    sm_cfg.par_merge_thresh=par_merge_thresh;
    sm_cfg.steal_factor=steal_factor; sm_cfg.rsort_thresh=rsort_thresh;
}

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
            if (!map[n]) {
                /* enc_v(13)=15 is the max 4-bit label; enc_v(14)=16 overflows.
                   With fs<=15 and max-degree-4 graphs this should never fire. */
                if (nxt > 15u) {
                    fprintf(stderr,
                        "FATAL: canon() label overflow at fs=%d — "
                        "more than 14 distinct path segments in frontier. "
                        "State encoding cannot represent this. "
                        "File a bug.\n", fs);
                    abort();
                }
                map[n] = nxt++;
            }
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

    int use_rsort = (curr->cnt >= sm_cfg.rsort_thresh);
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
    double     checkpoint_secs,
    int        step_limit        /* -1 = full run; >=0 = stop after this many steps */
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
    double t_ckpt_total = 0.0;  /* total time spent on checkpoint I/O, excluded from profile */

    if (verbose)
        fprintf(stderr,
            "step  vertex  fs  n_back  states_in  states_out"
            "  step_ms  cumul_ms\n");

    for (int step = start_step; step < n; step++) {
        int v = order[step];

        if (fs > MAX_FS_FAST) {
            *res_lo = *res_hi = UINT64_MAX;
            goto cleanup;
        }

        /* ---- A. Introduce v ------------------------------------------ */
        {
            int use_rsort_a = (curr->cnt >= sm_cfg.rsort_thresh);
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
                int use_rsort_b = (curr->cnt >= sm_cfg.rsort_thresh);
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
                int use_rsort_c = (curr->cnt >= sm_cfg.rsort_thresh);
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
            double dp_elapsed = t_now - t_start - t_ckpt_total;
            fprintf(stderr,
                "%4d  %6d  %2d  %6d  %9zu  %10zu  %8.1f  %8.1f\n",
                step, v, fs, n_back,
                states_in, curr->cnt,
                t_now - t_prev, dp_elapsed);
            t_prev = t_now;
        }

        if (checkpoint_path && checkpoint_path[0] && checkpoint_secs > 0) {
            double t_now2 = now_ms();
            if ((t_now2 - t_last_ckpt) / 1000.0 >= checkpoint_secs) {
                if (ckpt_save(checkpoint_path, n, order, step, fs,
                              frontier, total, curr)) {
                    double t_ckpt_end = now_ms();
                    double ckpt_ms = t_ckpt_end - t_now2;
                    t_ckpt_total += ckpt_ms;
                    if (verbose)
                        fprintf(stderr, "# Checkpoint saved at step %d (%zu states)"
                                " [ckpt_ms=%.1f, excluded from profile]\n",
                                step, curr->cnt, ckpt_ms);
                    /* Advance t_prev past checkpoint I/O so next step_ms
                       reflects DP time only, not disk write latency.       */
                    t_prev = t_ckpt_end;
                    t_last_ckpt = t_now2;
                }
            }
        }

        if (step_limit >= 0 && step >= step_limit) {
            double t_elapsed = now_ms() - t_start;
            uint64_t tmp; memcpy(&tmp, &t_elapsed, sizeof tmp);
            *res_lo = tmp;
            *res_hi = UINT64_MAX - 1;  /* sentinel: partial run (distinct from overflow) */
            goto cleanup;
        }
    }

cleanup:
    eh_free(curr);
    eh_free(nxt);
    free(fidx);

    /* Write path count only if neither sentinel is present.
       Overflow sentinel:     res_lo == res_hi == UINT64_MAX
       Partial-run sentinel:  res_hi == UINT64_MAX - 1        */
    if (*res_hi != UINT64_MAX && *res_hi != (UINT64_MAX - 1)) {
        *res_lo = (uint64_t) total;
        *res_hi = (uint64_t)(total >> 64);
    }
}

/* count_ham_paths_peh: PScan parallel backend removed (benchmarks showed
   it offers no advantage over EH+rsort: qsort inside workers costs as much
   as the compute it parallelises, and SM already does the job correctly
   with O(N) radix sort).  Keep the symbol so callers compile unchanged;
   internally just delegates to count_ham_paths_c.                        */
#include <pthread.h>   /* retained for potential future use */

void count_ham_paths_peh(
    int         n,
    const int  *order,
    const int  *pos,
    const int  *last_s,
    const int  *adj_off,
    const int  *adj_dat,
    uint64_t   *res_lo,
    uint64_t   *res_hi,
    int         verbose,
    const char *checkpoint_path,
    double      checkpoint_secs,
    int         step_limit
) {
    count_ham_paths_c(n, order, pos, last_s, adj_off, adj_dat,
                      res_lo, res_hi, verbose,
                      checkpoint_path, checkpoint_secs, step_limit);
}


/* ======================================================================
   SORT-MERGE DP IMPLEMENTATION
   -----------------------------------------------------------------------
   State table: SMEntry[] sorted by key.  Two alternating arrays (ping/pong).

   Per DP step:
     1. Introduce v   : sequential scan → new SMEntry[] (sorted by introduce())
     2. Fused sweep   : P worker threads, each maps a shard of the sorted
                        input; each worker runs buffered 8-bit radix sort on
                        its output into a CAPPED buffer; when the buffer is
                        full a sorted+deduped "run" is saved; at the end the
                        worker merges all its runs; the main thread then does
                        a P-way merge+dedup of the P per-worker streams.

   Buffer cap: sm_cfg.worker_cap entries per worker buf.  For large n where
   bpt = chunk*2^n_back would exceed this, the worker accumulates multiple
   sorted runs instead of one big allocation.  This bounds peak memory to
       curr + P * 2 * sm_cfg.worker_cap * 24B
   regardless of state count, preventing the OOM seen at n=61 step 37+.
   ====================================================================== */

#include <pthread.h>

typedef struct { u64 key; u64 val;  } SMEntry;  /* val: u64 safe for n<=~110 */

/* Fatal OOM handler — prints location and aborts cleanly.
   On macOS, malloc returns NULL under pressure rather than killing the
   process; without this the program crashes with a confusing SIGSEGV.  */
#define SM_OOM(bytes) do { \
    fprintf(stderr, "\nFATAL: out of memory allocating %zu bytes (%s:%d)\n" \
                    "       Reduce sm_cfg.worker_cap or use a machine with more RAM.\n", \
            (size_t)(bytes), __FILE__, __LINE__); \
    abort(); \
} while(0)
#define SM_ALLOC_CHECK(ptr, bytes) \
    do { if (!(ptr)) SM_OOM(bytes); } while(0)

/* ── Buffered 8-bit LSD radix sort ─────────────────────────────────── */
#define SM_RBUCKETS  256

typedef struct {
    SMEntry *slots; size_t cnt[SM_RBUCKETS]; size_t prefix[SM_RBUCKETS];
} RadixBuf;
static RadixBuf *sm_rbuf_alloc(void) {
    int rbs = sm_cfg.rbuf_size;
    RadixBuf *rb = (RadixBuf *)malloc(sizeof(RadixBuf));
    if (!rb) SM_OOM(sizeof(RadixBuf));
    rb->slots = (SMEntry *)malloc((size_t)SM_RBUCKETS*rbs*sizeof(SMEntry));
    if (!rb->slots) SM_OOM((size_t)SM_RBUCKETS*rbs*sizeof(SMEntry));
    memset(rb->cnt,0,sizeof(rb->cnt)); return rb;
}
static void sm_rbuf_free(RadixBuf *rb){if(rb){free(rb->slots);free(rb);}}

static void rbuf_pass(const SMEntry *src, size_t n, SMEntry *dst,
                      int shift, RadixBuf *rb) {
    int rbs = sm_cfg.rbuf_size;
    size_t cnt[SM_RBUCKETS] = {0};
    /* Use eh_hash(key) bits for bucket assignment.  Raw key bits are highly
       skewed (bits 56-63 take only 2 values for n=61), so the top passes of
       an unmodified LSD sort push all entries into 2 of 256 buckets.
       eh_hash is bijective, so sorting by eh_hash(key) is a valid total order:
       identical original keys always get the same hash, preserving dedup
       correctness; all internal merge/heap comparisons use the same hash.  */
    for (size_t i = 0; i < n; i++) cnt[(eh_hash(src[i].key) >> shift) & 0xff]++;
    size_t p = 0;
    for (int b = 0; b < SM_RBUCKETS; b++) {
        rb->prefix[b] = p; p += cnt[b]; rb->cnt[b] = 0;
    }
    for (size_t i = 0; i < n; i++) {
        int b = (eh_hash(src[i].key) >> shift) & 0xff;
        rb->slots[b*rbs + rb->cnt[b]++] = src[i];
        if (rb->cnt[b] == (size_t)rbs) {
            memcpy(dst + rb->prefix[b], rb->slots + b*rbs,
                   (size_t)rbs * sizeof(SMEntry));
            rb->prefix[b] += (size_t)rbs; rb->cnt[b] = 0;
        }
    }
    for (int b = 0; b < SM_RBUCKETS; b++)
        if (rb->cnt[b])
            memcpy(dst+rb->prefix[b], rb->slots+b*rbs, rb->cnt[b]*sizeof(SMEntry));
}

/* 8-pass LSD radix sort; result ends in 'a' (even pass count). */
static void sm_radix_sort(SMEntry *a, SMEntry *tmp, size_t n) {
    if (n < 2) return;
    RadixBuf *rb = sm_rbuf_alloc();
    for (int pass = 0; pass < 8; pass++) {
        SMEntry *src = (pass & 1) ? tmp : a;
        SMEntry *dst = (pass & 1) ? a   : tmp;
        rbuf_pass(src, n, dst, pass * 8, rb);
    }
    sm_rbuf_free(rb);
}

/* ── P-way merge with dedup (min-heap) ─────────────────────────────── */
typedef struct { const SMEntry *buf; size_t pos, len; } SMStream;
typedef struct { u64 key; int s; } SMHEntry;

static void sm_heap_sift(SMHEntry *h, int n, int i) {
    for (;;) {
        int m = i, l = 2*i+1, r = 2*i+2;
        if (l < n && eh_hash(h[l].key) < eh_hash(h[m].key)) m = l;
        if (r < n && eh_hash(h[r].key) < eh_hash(h[m].key)) m = r;
        if (m == i) break;
        SMHEntry t = h[i]; h[i] = h[m]; h[m] = t; i = m;
    }
}

/* ── Single-threaded P-way merge with dedup (used by sm_merge_runs and
   as the per-range kernel inside sm_parallel_merge) ──────────────────── */
static size_t sm_merge(SMStream *S, int P, SMEntry *out) {
    SMHEntry *h = (SMHEntry*)malloc(P * sizeof(SMHEntry));
    int hs = 0;
    for (int i = 0; i < P; i++)
        if (S[i].len > 0) { h[hs].key = S[i].buf[0].key; h[hs].s = i; hs++; }
    for (int i = hs/2-1; i >= 0; i--) sm_heap_sift(h, hs, i);
    size_t op = 0;
    while (hs > 0) {
        int s = h[0].s;
        SMEntry cur = S[s].buf[S[s].pos++];
        if (S[s].pos < S[s].len) {
            h[0].key = S[s].buf[S[s].pos].key; sm_heap_sift(h, hs, 0);
        } else { h[0] = h[--hs]; if (hs > 0) sm_heap_sift(h, hs, 0); }
        while (hs > 0 && h[0].key == cur.key) {
            int s2 = h[0].s;
            cur.val += S[s2].buf[S[s2].pos].val;
            S[s2].pos++;
            if (S[s2].pos < S[s2].len) {
                h[0].key = S[s2].buf[S[s2].pos].key; sm_heap_sift(h, hs, 0);
            } else { h[0] = h[--hs]; if (hs > 0) sm_heap_sift(h, hs, 0); }
        }
        out[op++] = cur;
    }
    free(h);
    return op;
}

/* ── Block-buffered K-way merge with dedup (sm_merge_bb) ───────────────
   Replaces sm_merge in latency-sensitive paths.  Each SMStream is wrapped
   in a sm_cfg.bb_buf-entry read buffer:

     sm_bb_fill() reads sm_cfg.bb_buf entries sequentially from the underlying
     array with a single memcpy — one DRAM round trip, sequential access,
     hardware-prefetcher-friendly.  The merge loop then reads from blk[]
     which is L1/L2 resident for sm_cfg.bb_buf≤32.

   Latency model (K streams, n elements total, T_DRAM=100ns):
     sm_merge:    n × T_DRAM                          (1 miss/element)
     sm_merge_bb: (n / sm_cfg.bb_buf) × T_DRAM            (1 miss per block)
   Speedup ≈ sm_cfg.bb_buf = 32×.  On large steps this converts the merge
   from latency-stall-bound (7% CPU) to bandwidth-bound (80%+ CPU).

   Buffer pool per call:  K × sm_cfg.bb_buf × 24B
     K=6  (parallel merge):  6×32×24 =  4.6 KB  → L1
     K=19 (ext merge runs): 19×32×24 = 14.6 KB  → L1
   Both fit in L1; no heap allocation needed for K≤512.        */


typedef struct {
    const SMEntry *base;   /* pointer into the sorted run/slice            */
    size_t         pos;    /* next unread index in base[]                  */
    size_t         len;    /* total entries in base[]                      */
    SMEntry       *blk;             /* bb_buf entries, pool-allocated       */
    int            bpos;   /* current read position in blk[]               */
    int            blen;   /* valid entries in blk[]                       */
} SMBufStr;

static inline void sm_bb_init(SMBufStr *s, const SMEntry *base, size_t len) {
    s->base = base; s->pos = 0; s->len = len; s->bpos = s->blen = 0;
}

/* Fill blk[] from the underlying array.  One sequential memcpy per call.  */
static inline void sm_bb_fill(SMBufStr *s) {
    size_t rem = s->len - s->pos;
    int n = (int)(rem < (size_t)sm_cfg.bb_buf ? rem : (size_t)sm_cfg.bb_buf);
    memcpy(s->blk, s->base + s->pos, (size_t)n * sizeof(SMEntry));
    s->pos += (size_t)n;
    s->bpos = 0;
    s->blen = n;
}

/* Is stream exhausted?  Triggers a refill if the buffer is empty.
   Returns 1 (true) only when both buffer and underlying array are empty.  */
static inline int sm_bb_empty(SMBufStr *s) {
    if (s->bpos < s->blen) return 0;
    if (s->pos  >= s->len) return 1;
    sm_bb_fill(s);
    return (s->blen == 0);
}

/* Current key, no side effects (stream must not be empty).               */
static inline u64 sm_bb_key(const SMBufStr *s) {
    return s->blk[s->bpos].key;
}

/* Consume and return current entry (stream must not be empty).           */
static inline SMEntry sm_bb_pop(SMBufStr *s) {
    return s->blk[s->bpos++];
}

/* Block-buffered K-way merge with dedup.  Same interface as sm_merge.    */
static size_t sm_merge_bb(SMStream *S, int K, SMEntry *out) {
    int bbs = sm_cfg.bb_buf;
    SMBufStr *B = (SMBufStr *)malloc((size_t)K * sizeof(SMBufStr));
    if (!B) SM_OOM((size_t)K * sizeof(SMBufStr));
    SMHEntry *h = (SMHEntry *)malloc((size_t)K * sizeof(SMHEntry));
    if (!h) SM_OOM((size_t)K * sizeof(SMHEntry));
    SMEntry *blk_pool = (SMEntry *)sm_alloc((size_t)K * bbs * sizeof(SMEntry));
    if (!blk_pool) SM_OOM((size_t)K * bbs * sizeof(SMEntry));

    /* Initialise buffered streams and pre-fill first block.              */
    for (int i = 0; i < K; i++) {
        B[i].base=S[i].buf; B[i].pos=0; B[i].len=S[i].len;
        B[i].blk=blk_pool+(size_t)i*bbs; B[i].bpos=0; B[i].blen=0;
        if (S[i].len > 0) sm_bb_fill(&B[i]);
    }

    /* Build initial min-heap (compare by eh_hash of raw key).            */
    int hs = 0;
    for (int i = 0; i < K; i++)
        if (!sm_bb_empty(&B[i])) { h[hs].key = sm_bb_key(&B[i]); h[hs].s = i; hs++; }
    for (int i = hs/2-1; i >= 0; i--) sm_heap_sift(h, hs, i);

    size_t op = 0;
    while (hs > 0) {
        int s = h[0].s;
        SMEntry cur = sm_bb_pop(&B[s]);           /* consume minimum entry */

        /* Advance stream s in the heap.                                  */
        if (!sm_bb_empty(&B[s])) {
            h[0].key = sm_bb_key(&B[s]);
            sm_heap_sift(h, hs, 0);
        } else {
            h[0] = h[--hs];
            if (hs > 0) sm_heap_sift(h, hs, 0);
        }

        /* Dedup: accumulate values from all streams sharing the same key. */
        while (hs > 0 && h[0].key == cur.key) {
            int s2 = h[0].s;
            cur.val += sm_bb_pop(&B[s2]).val;
            if (!sm_bb_empty(&B[s2])) {
                h[0].key = sm_bb_key(&B[s2]);
                sm_heap_sift(h, hs, 0);
            } else {
                h[0] = h[--hs];
                if (hs > 0) sm_heap_sift(h, hs, 0);
            }
        }
        out[op++] = cur;
    }

    free(B); free(h); sm_free(blk_pool);
    return op;
}

/* ── Parallel P-way merge via key-range splitting ───────────────────────
   Splits the P sorted input streams into K key ranges, assigns one merge
   thread per range.  Each thread merges its P sub-streams directly into a
   disjoint slice of the output array — no synchronisation needed during
   the merge itself.

   Algorithm:
     1. Sample S keys from each stream → pick K-1 splitter keys.
     2. Binary-search each stream for each splitter → lo/hi per (stream,range).
     3. Count outputs per range (upper bound = sum of sub-stream lengths).
     4. Allocate output array and compute write offsets.
     5. Launch K threads; each does sm_merge on its P sub-streams, writing
        into its slice.  Actual written count returned via MergeJob.out_cnt.
     6. Compact: prefix-sum the actual counts to produce the final array.

   Splitter selection: use the median of evenly-spaced samples across all
   streams, so the split is approximately balanced even for skewed key
   distributions.                                                          */

typedef struct {
    SMStream *subs;      /* P sub-streams for this range                   */
    int       P;
    SMEntry  *out;       /* write pointer into the output array            */
    size_t    out_cap;   /* allocated capacity (upper bound)               */
    size_t    out_cnt;   /* actual entries written (set by thread)         */
} SMParMergeJob;

static void *sm_par_merge_thread(void *arg) {
    SMParMergeJob *j = (SMParMergeJob *)arg;
    /* Large contiguous sub-stream slices: hardware prefetcher handles
       sequential access; sm_merge_bb overhead exceeds benefit here. */
    j->out_cnt = sm_merge(j->subs, j->P, j->out);
    return NULL;
}

/* Binary search: index of first entry in hash-sorted array a[0..n) whose
   eh_hash(key) >= target_hash.  Used by sm_parallel_merge to split streams
   at hash-space splitter values.                                           */
static size_t sm_lower_bound(const SMEntry *a, size_t n, u64 target_hash) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (eh_hash(a[mid].key) < target_hash) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* Parallel merge: merges P sorted streams into caller-provided buffer 'out'
   (capacity out_cap entries).  Sets *out_len to actual entries written.
   Falls back to single-threaded sm_merge when K=1 or input is small.
   Writing directly into the caller's buffer avoids the extra 16GB malloc
   that would otherwise coexist with W[i].out and nxt->data at step 39. */
#define SM_SAMPLES_PER_STREAM 64

static size_t sm_parallel_merge(SMStream *S, int P, int K,
                                 SMEntry *out, size_t out_cap) {
    /* Compute total input size */
    size_t total = 0;
    for (int i = 0; i < P; i++) total += S[i].len;

    /* Fall back to single-threaded for small inputs */
    if (K <= 1 || total < sm_cfg.par_merge_thresh)
        return sm_merge(S, P, out);

    /* ── Step 1: collect sample keys ──────────────────────────────── */
    int n_samples = P * SM_SAMPLES_PER_STREAM;
    u64 *samples  = (u64 *)malloc(n_samples * sizeof(u64));
    int  ns = 0;
    for (int i = 0; i < P; i++) {
        size_t len = S[i].len;
        if (len == 0) continue;
        int s = SM_SAMPLES_PER_STREAM;
        for (int k = 0; k < s; k++) {
            size_t idx = (size_t)k * (len > 1 ? len - 1 : 0) / (s > 1 ? s - 1 : 1);
            /* Sample eh_hash(key) values: the hash is uniformly distributed
               even when raw keys cluster in just 2-4 top-byte values, giving
               accurate quantile estimates and balanced thread work.         */
            samples[ns++] = eh_hash(S[i].buf[idx].key);
        }
    }
    /* Insertion sort on samples (tiny array) */
    for (int i = 1; i < ns; i++) {
        u64 v = samples[i]; int j = i - 1;
        while (j >= 0 && samples[j] > v) { samples[j+1] = samples[j]; j--; }
        samples[j+1] = v;
    }
    /* Pick K-1 splitters evenly from sorted samples */
    u64 *splitters = (u64 *)malloc((K - 1) * sizeof(u64));
    for (int j = 0; j < K - 1; j++) {
        int idx = (int)((size_t)(j + 1) * ns / K);
        if (idx >= ns) idx = ns - 1;
        splitters[j] = samples[idx];
    }
    free(samples);
    /* Ensure strictly increasing */
    for (int j = 1; j < K - 1; j++)
        if (splitters[j] <= splitters[j-1]) splitters[j] = splitters[j-1] + 1;

    /* ── Step 2: binary-search each stream for each splitter ─────── */
    size_t *boundaries = (size_t *)malloc((size_t)P * (K + 1) * sizeof(size_t));
#define BND(i,j) boundaries[(size_t)(i)*(K+1)+(j)]
    for (int i = 0; i < P; i++) {
        BND(i, 0) = 0;
        for (int j = 0; j < K - 1; j++)
            BND(i, j+1) = sm_lower_bound(S[i].buf, S[i].len, splitters[j]);
        BND(i, K) = S[i].len;
    }
    free(splitters);

    /* ── Step 3: compute per-range capacities and write offsets ───── */
    size_t *offsets = (size_t *)malloc((K + 1) * sizeof(size_t));
    offsets[0] = 0;
    for (int j = 0; j < K; j++) {
        size_t cap = 0;
        for (int i = 0; i < P; i++) cap += BND(i, j+1) - BND(i, j);
        offsets[j+1] = offsets[j] + cap;
    }
    /* offsets[K] <= out_cap since total input <= out_cap */

    /* ── Step 4: launch K merge threads writing into out directly ── */
    SMParMergeJob *jobs    = (SMParMergeJob *)malloc(K * sizeof(SMParMergeJob));
    SMStream      *all_sub = (SMStream *)malloc((size_t)K * P * sizeof(SMStream));
    pthread_t     *threads = (pthread_t *)malloc(K * sizeof(pthread_t));

    for (int j = 0; j < K; j++) {
        SMStream *sub = &all_sub[(size_t)j * P];
        for (int i = 0; i < P; i++) {
            size_t lo = BND(i, j), hi = BND(i, j+1);
            sub[i] = (SMStream){ S[i].buf + lo, 0, hi - lo };
        }
        jobs[j] = (SMParMergeJob){
            .subs    = sub,
            .P       = P,
            .out     = out + offsets[j],
            .out_cap = offsets[j+1] - offsets[j],
            .out_cnt = 0,
        };
        pthread_create(&threads[j], NULL, sm_par_merge_thread, &jobs[j]);
    }
    for (int j = 0; j < K; j++) pthread_join(threads[j], NULL);
#undef BND
    free(boundaries);

    /* ── Step 5: compact gaps from deduplication ─────────────────── */
    size_t write_pos = 0, out_len = 0;
    for (int j = 0; j < K; j++) {
        size_t cnt = jobs[j].out_cnt;
        if (cnt == 0) continue;
        SMEntry *src = out + offsets[j];
        if (src != out + write_pos)
            memmove(out + write_pos, src, cnt * sizeof(SMEntry));
        write_pos += cnt;
        out_len   += cnt;
    }

    free(jobs); free(all_sub); free(threads); free(offsets);
    return out_len;
}

/* ── SM table ───────────────────────────────────────────────────────── */
typedef struct { SMEntry *data; size_t cnt, cap; } SMTab;

static SMTab *smtab_alloc(size_t cap) {
    SMTab *t = (SMTab*)calloc(1, sizeof(SMTab));
    t->cap = cap < 1 ? 1024 : cap;
    t->data = (SMEntry*)sm_alloc(t->cap * sizeof(SMEntry));
    return t;
}
static void smtab_free(SMTab *t) { sm_free(t->data); free(t); }
static void smtab_ensure(SMTab *t, size_t needed) {
    if (needed <= t->cap) return;
    size_t old_cap   = t->cap;
    size_t old_bytes = old_cap * sizeof(SMEntry);
    if (t->cap == 0) t->cap = 1024;
    while (t->cap < needed) t->cap *= 2;
    size_t new_bytes = t->cap * sizeof(SMEntry);
    /* sm_realloc only handles same-zone growth (both small or both large).
       When crossing the SM_LARGE_THRESHOLD (e.g. malloc 768KB → mmap 1.5MB),
       we must do the copy explicitly because sm_realloc doesn't know the old
       malloc size and cannot memcpy safely.  sm_alloc + memcpy + sm_free is
       always correct and is inplace on Linux (mremap) for large→large.      */
    SMEntry *np = (SMEntry *)sm_alloc(new_bytes);
    SM_ALLOC_CHECK(np, new_bytes);
    if (t->data && old_bytes > 0)
        memcpy(np, t->data, old_bytes);
    sm_free(t->data);
    t->data = np;
}

/* ── SM introduce ───────────────────────────────────────────────────── */
/* In-place parallel introduce: transform tab->data[i].key in-place.
   introduce(key, fs) = key | (enc_v(0) << (4*fs))
   It only ORs in bits at position fs and never touches bits 0..fs-1,
   so each entry can be updated independently without reading any other
   entry.  In-place operation halves the peak memory vs a copy:
     - Copy approach: curr(si) + nxt(si) = 2×si simultaneously
     - In-place:      curr(si) only — nxt is allocated later in sm_fused_sweep
   At n=75 step 57 (13.3B states = 303GB), this saves 303GB.            */

typedef struct { SMEntry *data; size_t lo, hi; int fs; } SMIntroJob;

static void *sm_intro_thread(void *arg) {
    SMIntroJob *j = (SMIntroJob *)arg;
    u64 mask = (u64)enc_v(0) << (4 * j->fs);
    for (size_t i = j->lo; i < j->hi; i++)
        j->data[i].key |= mask;
    return NULL;
}

static void sm_introduce(SMTab *tab, int fs) {
    int P = sm_cfg.nthreads;
    if (P <= 1 || tab->cnt < 65536) {
        u64 mask = (u64)enc_v(0) << (4 * fs);
        for (size_t i = 0; i < tab->cnt; i++)
            tab->data[i].key |= mask;
        return;
    }
    SMIntroJob *jobs = (SMIntroJob *)malloc((size_t)P * sizeof(SMIntroJob));
    pthread_t  *thds = (pthread_t  *)malloc((size_t)P * sizeof(pthread_t));
    SM_ALLOC_CHECK(jobs, (size_t)P * sizeof(SMIntroJob));
    SM_ALLOC_CHECK(thds, (size_t)P * sizeof(pthread_t));
    size_t chunk = (tab->cnt + (size_t)P - 1) / (size_t)P;
    for (int i = 0; i < P; i++) {
        size_t lo = (size_t)i * chunk;
        size_t hi = lo + chunk < tab->cnt ? lo + chunk : tab->cnt;
        jobs[i] = (SMIntroJob){ tab->data, lo, hi, fs };
        pthread_create(&thds[i], NULL, sm_intro_thread, &jobs[i]);
    }
    for (int i = 0; i < P; i++) pthread_join(thds[i], NULL);
    free(jobs); free(thds);
}

/* ── SM worker (capped buffer + run accumulation) ───────────────────── */

/* Per-worker buffer cap (entries).  Bounds peak RAM to
   curr + P * 2 * sm_cfg.worker_cap * 24B regardless of state count.
   32M entries = 768 MB per worker buf; total 6*2*768MB = 9.2 GB.       */

typedef struct { SMEntry *data; size_t len; size_t alloc_size; } SMRun;

/* Align to cache line (128B on Apple Silicon, 64B on x86).
   Without this, all workers share 2-3 cache lines, and mid-compute
   flushes by one worker invalidate neighbouring workers' buf_len fields,
   causing 3-4× slowdown specifically at n_runs=2 steps.               */
#if defined(__APPLE__) && defined(__aarch64__)
#  define SM_CACHE_ALIGN __attribute__((aligned(128)))
#else
#  define SM_CACHE_ALIGN __attribute__((aligned(64)))
#endif

typedef struct SM_CACHE_ALIGN {
    /* worker scratch (reused across runs) */
    SMEntry *buf;   /* output accumulation buffer, sm_cfg.worker_cap entries */
    SMEntry *tmp;   /* radix sort scratch,         sm_cfg.worker_cap entries */
    size_t   buf_len;
    /* completed sorted+deduped runs */
    int      ext_runs;
    SMRun   *runs;
    int      n_runs, runs_cap;
    int      run_fd;
    void    *run_map;
    size_t   run_capacity;
    size_t   run_fd_off;
    size_t  *run_lens;
} SMWorkerState;

/* ── ext_runs backing store helpers ─────────────────────────────────── */
#ifdef __APPLE__
#  define SM_USE_SHM 1
#else
#  define SM_USE_SHM 0
#endif
/* MAP_ANONYMOUS / MAP_ANON: normalise to MAP_ANONYMOUS.
   macOS (_DARWIN_C_SOURCE): MAP_ANON defined, MAP_ANONYMOUS may not be.
   Linux: MAP_ANONYMOUS defined, MAP_ANON may not be.                   */
#ifndef MAP_ANONYMOUS
#  ifdef MAP_ANON
#    define MAP_ANONYMOUS MAP_ANON
#  else
#    define MAP_ANONYMOUS 0x20   /* Linux numeric fallback */
#  endif
#endif

/* Open backing store for a worker's run data.
   Strategy (in order of preference):
     1. MAP_ANON mmap (macOS + Linux): anonymous VM, RAM bandwidth,
        no filesystem, no ftruncate size limit, lazy physical allocation.
        fd=-2 sentinel signals map-only mode to sm_ext_write/read/close.
     2. shm_open + ftruncate + mmap (macOS fallback for small sizes):
        POSIX shm, also RAM-backed but has per-object size limits.
     3. tmpfile (Linux / final fallback): tmpfs on Linux = RAM bandwidth;
        APFS on macOS = catastrophically slow, avoid if possible.

   We prefer MAP_ANON because it works for any size, has no quota
   limits, and avoids all filesystem involvement on macOS.              */
static int sm_ext_open(size_t capacity_bytes, void **map_out) {
    *map_out = NULL;
    if (capacity_bytes == 0) capacity_bytes = sizeof(SMEntry);

    /* 1. Try MAP_ANON — works on both macOS and Linux, no fd needed.  */
    void *m = mmap(NULL, capacity_bytes,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    if (m != MAP_FAILED) {
        *map_out = m;
        return -2;          /* sentinel: map-only mode, no fd           */
    }

#if SM_USE_SHM
    /* 2. Try shm_open (macOS only, works for small-ish sizes).        */
    static volatile int shm_seq = 0;
    char name[64];
    int seq = __sync_fetch_and_add(&shm_seq, 1);
    snprintf(name, sizeof(name), "/ham_dp_%d_%d", (int)getpid(), seq);
    int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    shm_unlink(name);
    if (fd >= 0) {
        if (ftruncate(fd, (off_t)capacity_bytes) == 0) {
            void *sm = mmap(NULL, capacity_bytes,
                            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (sm != MAP_FAILED) { *map_out = sm; return fd; }
        }
        close(fd);
    }
#endif

    /* 3. Fall back to tmpfile (Linux tmpfs = fast; macOS APFS = slow).*/
    FILE *f = tmpfile();
    return f ? fileno(f) : -1;
}

/* Write n entries at byte offset *off; advance *off.
   map != NULL → direct memcpy into the mmap region (macOS path).
   map == NULL → pwrite to fd (Linux path).                             */
static void sm_ext_write(int fd, void *map, size_t *off,
                          const SMEntry *data, size_t n) {
    size_t bytes = n * sizeof(SMEntry);
    if (map) {
        memcpy((char *)map + *off, data, bytes);
    } else {
        ssize_t w = pwrite(fd, data, bytes, (off_t)*off);
        (void)w;
    }
    *off += bytes;
}

/* Read n entries from byte offset off.                                 */
static void sm_ext_read(int fd, const void *map, size_t off,
                         SMEntry *data, size_t n) {
    if (map) {
        memcpy(data, (const char *)map + off, n * sizeof(SMEntry));
    } else {
        ssize_t r = pread(fd, data, n * sizeof(SMEntry), (off_t)off);
        (void)r;
    }
}

/* Close and release a backing store.                                   */
static void sm_ext_close(int fd, void *map, size_t capacity_bytes) {
    if (map && map != MAP_FAILED)
        munmap(map, capacity_bytes);
    if (fd >= 0) close(fd);
    /* fd == -2: MAP_ANON mode, map already released above; no fd.     */
}

/* ── Streaming ext-mode merge ─────────────────────────────────────────
   Merges n_runs sorted runs (stored in a file or mmap) into a single
   sorted+deduped array, reading each run through a small buffer.
   Peak extra RAM = n_runs × sm_cfg.ext_stream_buf × 24B (e.g. 31 × 24MB = 744MB)
   vs loading all runs simultaneously (e.g. 2.7 GB for step 38).
   Large sequential reads → OS readahead → near-SSD-peak bandwidth.     */

typedef struct {
    int      fd;
    void    *map;
    size_t   run_off;    /* byte offset of run start in file             */
    size_t   run_len;    /* total entries in run                         */
    size_t   run_pos;    /* entries consumed from file/map so far        */
    SMEntry *buf;
    size_t   buf_len;
    size_t   buf_pos;
} SMExtStream;

static void sm_ext_stream_refill(SMExtStream *s) {
    if (s->run_pos >= s->run_len) { s->buf_len = 0; return; }
    size_t remaining = s->run_len - s->run_pos;
    size_t n = remaining < (size_t)sm_cfg.ext_stream_buf ? remaining
                                                      : (size_t)sm_cfg.ext_stream_buf;
    size_t off = s->run_off + s->run_pos * sizeof(SMEntry);
    if (s->map)
        memcpy(s->buf, (const char *)s->map + off, n * sizeof(SMEntry));
    else
        pread(s->fd, s->buf, n * sizeof(SMEntry), (off_t)off);
    s->buf_pos = 0;
    s->buf_len = n;
    s->run_pos += n;
}

/* Advance stream s in the heap by one entry; refill its buffer if empty. */
static inline void sm_ext_advance(SMHEntry *h, int *hs,
                                   SMExtStream *S, int s) {
    if (S[s].buf_pos < S[s].buf_len) {
        h[0].key = S[s].buf[S[s].buf_pos].key;
        sm_heap_sift(h, *hs, 0);
    } else {
        sm_ext_stream_refill(&S[s]);
        if (S[s].buf_len > 0) {
            h[0].key = S[s].buf[0].key;
            sm_heap_sift(h, *hs, 0);
        } else {
            h[0] = h[--(*hs)];
            if (*hs > 0) sm_heap_sift(h, *hs, 0);
        }
    }
}

/* ── Work-stealing shared micro-chunk queue ─────────────────────────── */

typedef struct {
    volatile size_t next;
    size_t          total;
    size_t          mchunk;
} SMWorkQueue;

typedef struct {
    const SMEntry   *in;
    size_t           lo, hi;
    int              fs, v_idx;
    const int       *widxs;
    int              n_back;
    const int       *elim_idxs_desc;
    int              n_elim;
    int              step, n;
    u128            *total;
    pthread_mutex_t *total_mu;
    SMWorkerState   *ws;
    SMEntry         *out;
    size_t           out_len;
    size_t           out_size;   /* bytes allocated for out (for sm_free) */
    double           t_compute_ms;
    double           t_flush_ms;
    double           t_merge_ms;
    size_t           raw_out;
} SMWorker;

/* ── sm_flush_run: sort+dedup buf, then store as a run ─────────────── */
static void sm_flush_run(SMWorkerState *ws) {
    if (ws->buf_len == 0) return;
    sm_radix_sort(ws->buf, ws->tmp, ws->buf_len);
    /* dedup in-place */
    size_t n = ws->buf_len, o = 0;
    for (size_t i = 0; i < n; i++) {
        if (o > 0 && ws->buf[o-1].key == ws->buf[i].key)
            ws->buf[o-1].val += ws->buf[i].val;
        else
            ws->buf[o++] = ws->buf[i];
    }
    ws->buf_len = 0;

    if (ws->ext_runs) {
        sm_ext_write(ws->run_fd, ws->run_map, &ws->run_fd_off, ws->buf, o);
        if (ws->n_runs == ws->runs_cap) {
            ws->runs_cap = ws->runs_cap ? ws->runs_cap * 2 : 16;
            ws->run_lens = (size_t*)realloc(ws->run_lens,
                                            ws->runs_cap * sizeof(size_t));
        }
        ws->run_lens[ws->n_runs++] = o;
    } else {
        /* RAM mode: copy run data using sm_alloc (bypasses xzone on macOS). */
        size_t run_bytes = o * sizeof(SMEntry);
        SMEntry *run_data = (SMEntry*)sm_alloc(run_bytes);
        memcpy(run_data, ws->buf, run_bytes);
        if (ws->n_runs == ws->runs_cap) {
            ws->runs_cap = ws->runs_cap ? ws->runs_cap * 2 : 4;
            ws->runs = (SMRun*)realloc(ws->runs,
                                       ws->runs_cap * sizeof(SMRun));
        }
        ws->runs[ws->n_runs++] = (SMRun){ run_data, o, run_bytes };
    }
}

/* ── Bucketed merge helpers ─────────────────────────────────────────────
   Split runs by top 8 bits of key (256 buckets).  Entries in different
   buckets have different keys and can never merge, so each bucket is an
   independent sub-problem.  This shrinks the heap merge working set by
   256×, pulling it from DRAM into L2 cache.

   For n_runs=2, working set drops from 480 MB → 1.9 MB (L2).
   For n=61 ext steps (n_runs=19), drops from 8 GB → 32 MB (L2/L3).    */

/* Hash-based bucket of a key (0..255).
   Keys have bits 56-61 always zero and only bits 62-63 varying in the high
   byte, so raw top-byte extraction gives only 2-4 distinct buckets out of
   256.  Using eh_hash first mixes all key bits uniformly, producing a
   near-perfect 256-way split and enabling effective cache reuse in the
   bucketed merge and balanced work distribution in sm_parallel_merge.
   eh_hash is bijective, so bucket membership is stable and dedup by original
   key remains correct.                                                    */
static inline int sm_bucket(u64 key) { return (int)(eh_hash(key) >> 56); }

/* Binary search: index of first entry in hash-sorted array a[0..n) whose
   hash-bucket is >= b.  Since sm_bucket = top byte of eh_hash, and the array
   is sorted by eh_hash(key), bucket values are monotone non-decreasing.    */
static size_t sm_bucket_lower(const SMEntry *a, size_t n, int b) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (sm_bucket(a[mid].key) < b) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* ── sm_merge_runs: merge all runs → single sorted+deduped array ───── */
static SMEntry *sm_merge_runs(SMWorkerState *ws, size_t *out_len) {
    if (ws->buf_len > 0) sm_flush_run(ws);

    if (ws->ext_runs) {
        /* ── Ext path: streaming heap merge ────────────────────────── */
        /* Each run is read through a SM_EXT_STREAM_BUF-entry buffer.
           Peak extra RAM = n_runs × sm_cfg.ext_stream_buf × 24B (small).
           MAP_ANON gives RAM bandwidth; sequential access + hardware
           prefetch makes this fast without bucketing.                  */
        if (ws->n_runs == 0) { *out_len = 0; return NULL; }

        int     n_runs = ws->n_runs;
        size_t  total  = 0;
        for (int i = 0; i < n_runs; i++) total += ws->run_lens[i];
        SMEntry *out = (SMEntry *)sm_alloc(total * sizeof(SMEntry));
        SM_ALLOC_CHECK(out, total * sizeof(SMEntry));

        if (n_runs == 1) {
            sm_ext_read(ws->run_fd, ws->run_map, 0, out, ws->run_lens[0]);
            *out_len = ws->run_lens[0];
            return out;
        }

        SMExtStream *S = (SMExtStream *)malloc(n_runs * sizeof(SMExtStream));
        SM_ALLOC_CHECK(S, n_runs * sizeof(SMExtStream));
        SMHEntry    *h = (SMHEntry *)malloc(n_runs * sizeof(SMHEntry));
        SM_ALLOC_CHECK(h, n_runs * sizeof(SMHEntry));

        size_t file_off = 0;
        for (int i = 0; i < n_runs; i++) {
            S[i] = (SMExtStream){
                .fd = ws->run_fd, .map = ws->run_map,
                .run_off = file_off, .run_len = ws->run_lens[i],
                .run_pos = 0, .buf_len = 0, .buf_pos = 0,
                .buf = (SMEntry *)malloc(sm_cfg.ext_stream_buf * sizeof(SMEntry)),
            };
            SM_ALLOC_CHECK(S[i].buf, sm_cfg.ext_stream_buf * sizeof(SMEntry));
            file_off += ws->run_lens[i] * sizeof(SMEntry);
            sm_ext_stream_refill(&S[i]);
        }

        int hs = 0;
        for (int i = 0; i < n_runs; i++)
            if (S[i].buf_len > 0) { h[hs].key = S[i].buf[0].key; h[hs].s = i; hs++; }
        for (int i = hs/2-1; i >= 0; i--) sm_heap_sift(h, hs, i);

        size_t op = 0;
        while (hs > 0) {
            int s = h[0].s;
            SMEntry cur = S[s].buf[S[s].buf_pos++];
            sm_ext_advance(h, &hs, S, s);
            while (hs > 0 && h[0].key == cur.key) {
                int s2 = h[0].s;
                cur.val += S[s2].buf[S[s2].buf_pos++].val;
                sm_ext_advance(h, &hs, S, s2);
            }
            out[op++] = cur;
        }

        for (int i = 0; i < n_runs; i++) free(S[i].buf);
        free(S); free(h);
        *out_len = op;
        return out;

    } else {
        /* ── RAM path: bucketed in-memory merge ────────────────────── */
        if (ws->n_runs == 0) { *out_len = 0; return NULL; }
        if (ws->n_runs == 1) {
            *out_len = ws->runs[0].len;
            SMEntry *d = ws->runs[0].data;
            ws->runs[0].data = NULL;
            return d;
        }

        int     n_runs = ws->n_runs;
        size_t  total  = 0;
        for (int i = 0; i < n_runs; i++) total += ws->runs[i].len;
        size_t out_bytes = total * sizeof(SMEntry);
        SMEntry  *out = (SMEntry *)sm_alloc(out_bytes);
        SM_ALLOC_CHECK(out, out_bytes);

        /* For each run, find the 257 bucket boundary indices via
           binary search.  bounds[i][b] = first index in run i with
           top byte >= b.  This costs O(n_runs × 256 × log(run_size))
           ≈ negligible.                                                */
        size_t (*bounds)[257] = malloc(n_runs * sizeof(*bounds));
        SM_ALLOC_CHECK(bounds, n_runs * sizeof(*bounds));
        for (int i = 0; i < n_runs; i++) {
            const SMEntry *d = ws->runs[i].data;
            size_t len       = ws->runs[i].len;
            bounds[i][0]     = 0;
            for (int b = 1; b <= 256; b++)
                bounds[i][b] = (b <= 255)
                    ? sm_bucket_lower(d, len, b)
                    : len;
        }

        /* Allocate stream descriptors (one per run); reused across buckets. */
        SMStream *S = (SMStream *)malloc(n_runs * sizeof(SMStream));

        size_t write_pos = 0;
        for (int b = 0; b < 256; b++) {
            /* Set each stream to its bucket-b slice. */
            int active = 0;
            for (int i = 0; i < n_runs; i++) {
                size_t lo = bounds[i][b], hi = bounds[i][b+1];
                S[i] = (SMStream){ ws->runs[i].data + lo, 0, hi - lo };
                if (hi > lo) active++;
            }
            if (active == 0) continue;
            /* Small N-way merge for this bucket. */
            write_pos += sm_merge(S, n_runs, out + write_pos);
        }

        free(S); free(bounds);
        for (int i = 0; i < n_runs; i++)
            sm_free(ws->runs[i].data);
        *out_len = write_pos;
        return out;
    }
}

/* Atomically claim the next micro-chunk. */
static size_t sm_wq_claim(SMWorkQueue *wq) {
    return __sync_fetch_and_add(&wq->next, 1);
}

static void *sm_worker_runs(void *arg) {
    SMWorker      *w  = (SMWorker*)arg;
    SMWorkerState *ws = w->ws;
    int n_subsets = 1 << w->n_back;
    u128 local_total = 0;
    size_t raw_out = 0;

    double t0 = now_ms(), t_flush_acc = 0.0;

    for (size_t i = w->lo; i < w->hi; i++) {
        u64  base = w->in[i].key;
        u64  cnt  = w->in[i].val;
        for (int S = 0; S < n_subsets; S++) {
            u64 nk = base; int valid = 1; int nc_inc = 0;
            for (int j = 0; j < w->n_back && valid; j++) {
                if (!(S & (1 << j))) continue;
                u64 nk2 = apply_edge(nk, w->fs, w->v_idx, w->widxs[j], &nc_inc);
                if (!nk2) { valid = 0; break; }
                nk = nk2;
            }
            if (!valid) continue;
            if (w->n_elim > 0) {
                nk = apply_elim_seq(nk, w->fs, w->elim_idxs_desc, w->n_elim,
                                    w->step, w->n, cnt, &local_total);
                if (!nk) continue;
            }
            ws->buf[ws->buf_len++] = (SMEntry){ nk, cnt };
            raw_out++;
            if (ws->buf_len == sm_cfg.worker_cap) {
                double tf0 = now_ms();
                sm_flush_run(ws);
                t_flush_acc += now_ms() - tf0;
            }
        }
    }
    if (ws->buf_len > 0) {
        double tf0 = now_ms();
        sm_flush_run(ws);
        t_flush_acc += now_ms() - tf0;
    }

    double t1 = now_ms();
    w->t_compute_ms = (t1 - t0) - t_flush_acc;
    w->t_flush_ms   = t_flush_acc;
    w->raw_out      = raw_out;

    sm_free(ws->buf); ws->buf = NULL;
    sm_free(ws->tmp); ws->tmp = NULL;

    if (local_total) {
        pthread_mutex_lock(w->total_mu);
        *w->total += local_total;
        pthread_mutex_unlock(w->total_mu);
    }
    return NULL;
}


/* ── sm_pairwise_merge_into ─────────────────────────────────────────────────
   Merge P sorted+deduped arrays into nxt using a binary-tree reduction.
   Each level processes pairs sequentially, freeing inputs immediately and
   reallocing each result to its actual (deduped) size.

   Memory profile for P=6, step 38 (raw_total=23.7 GB, deduped=12.4 GB):

     Level 0 – three sequential 2-way merges of W pairs:
       Peak per pair: curr + W_all + T_pair_alloc ≈ 7.7+23.7+7.9+OS = 43.3 GB
       After each pair: W pair freed, T shrunk to actual size (~4.4 GB)
       After level 0:   curr + T01+T23+T45 ≈ 7.7+3×4.4+OS = 24.0 GB   ← fits!

     Level 1 – merge T01+T23 → T0123:
       Peak: curr + T01(4.4) + T23(4.4) + T45(4.4) + T0123(8.8) + OS ≈ 33.7 GB ← fits!
       After T0123 shrunk to actual: curr + T45(4.4) + T0123(8.3) + OS ≈ 24.1 GB

     Level 2 – final merge T0123+T45 → nxt (smtab_ensure for actual size):
       Peak: curr + T0123(8.3) + T45(4.4) + nxt(12.4) + OS ≈ 36.8 GB   ← near fit!

   Compare current code: curr + W_all + nxt_for_raw = 59.1 GB peak.

   Correctness: 2-way merge with dedup is associative for aggregation — the
   same key appearing in multiple input arrays gets its values summed at each
   merge step, giving the same result as a flat P-way merge.                 */
static void sm_pairwise_merge_into(SMEntry **arrs, size_t *lens, int P,
                                    SMTab *nxt) {
    if (P == 0) { nxt->cnt = 0; return; }

    int cur_P = P;
    while (cur_P > 1) {
        int new_P = (cur_P + 1) / 2;

        for (int i = 0; i < cur_P / 2; i++) {
            size_t cap = lens[2*i] + lens[2*i+1];
            SMStream S[2] = {
                { arrs[2*i],   0, lens[2*i]   },
                { arrs[2*i+1], 0, lens[2*i+1] },
            };
            size_t cnt;

            /* Save input pointers BEFORE any assignment to arrs[i].
               When i==0, arrs[i] and arrs[2*i] are the same slot; assigning
               arrs[i]=merged would overwrite arrs[0] and then free(arrs[0])
               would free the result instead of the original input.            */
            SMEntry *in_left  = arrs[2*i];
            SMEntry *in_right = arrs[2*i+1];

            if (cur_P == 2) {
                /* Final merge: allocate nxt for exact pair size, not raw_total.
                   This is the key memory saving: nxt = deduped bytes, not raw. */
                smtab_ensure(nxt, cap + 1);
                cnt      = sm_merge_bb(S, 2, nxt->data);
                nxt->cnt = cnt;
            } else {
                SMEntry *merged = (SMEntry*)sm_alloc(cap * sizeof(SMEntry));
                SM_ALLOC_CHECK(merged, cap * sizeof(SMEntry));
                cnt = sm_merge_bb(S, 2, merged);
                /* Shrink to actual (deduped) size: on macOS the libmalloc
                   large-allocation path returns physical pages to the OS,
                   reducing RSS before the next pair is allocated.             */
                if (cnt < cap) {
                    SMEntry *sh = (SMEntry*)sm_realloc(merged, cnt * sizeof(SMEntry));
                    if (sh) merged = sh;
                }
                arrs[i] = merged;
                lens[i] = cnt;
            }

            /* Free originals via saved pointers (safe against aliasing).
               Null-out rules:
                 Final merge (cur_P==2): result is in nxt, so NULL both
                   input slots unconditionally.
                 Non-final (else branch): result is in arrs[i].  When i==0,
                   arrs[i] and arrs[2*i] are the SAME slot — nulling arrs[2*i]
                   would destroy the result we just stored there.  Skip it.
                   arrs[2*i+1] is always a distinct slot (2i+1 > i).         */
            sm_free(in_left);
            sm_free(in_right);
            if (cur_P == 2 || i != 0) arrs[2*i] = NULL;
            arrs[2*i+1] = NULL;
        }

        /* Carry forward the odd element when cur_P is odd. */
        if (cur_P & 1) {
            arrs[new_P - 1] = arrs[cur_P - 1];
            lens[new_P - 1] = lens[cur_P - 1];
            arrs[cur_P - 1] = NULL;  /* prevent double-free */
        }
        cur_P = new_P;
    }

    /* cur_P == 1 only when P was 1 to begin with (while loop never ran). */
    if (arrs[0] != NULL) {
        smtab_ensure(nxt, lens[0] + 1);
        nxt->cnt = lens[0];
        if (arrs[0] != nxt->data)
            memcpy(nxt->data, arrs[0], lens[0] * sizeof(SMEntry));
        sm_free(arrs[0]);
        arrs[0] = NULL;
    }
}


/* Forward declaration — defined below. */
static size_t sm_global_ext_merge(SMWorkerState *WS, int P,
                                   SMTab *nxt, size_t raw_total,
                                   double *t_merge_ms_out);

/* ── sm_global_ext_merge_par helpers ─────────────────────────────────────
   Binary search within one ext stream for the first entry whose
   eh_hash(key) >= target_hash.  Uses random access into the mmap region
   (or pread for file-backed stores).  O(log run_len) random reads, each
   fetching sizeof(SMEntry)=16 bytes; since MAP_ANON data is RAM-resident
   the latency is L3-cache-miss level, not disk I/O.                     */
static size_t sm_ext_stream_lower_bound(
    int fd, void *map, size_t run_off_bytes, size_t run_len_entries,
    u64 target_hash)
{
    size_t lo = 0, hi = run_len_entries;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        SMEntry e;
        size_t off = run_off_bytes + mid * sizeof(SMEntry);
        if (map)
            memcpy(&e, (const char *)map + off, sizeof(SMEntry));
        else
            pread(fd, &e, sizeof(SMEntry), (off_t)off);
        if (eh_hash(e.key) < target_hash) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* Per-thread job for sm_global_ext_merge_par.                           */
typedef struct {
    SMExtStream *streams;    /* K pre-initialised streams for this range */
    int          K;          /* number of streams (= total_streams)       */
    size_t       dyn_buf;    /* per-stream read-ahead buffer (entries)    */
    SMEntry     *out;        /* output slice (pre-allocated by caller)    */
    size_t       out_cap;    /* capacity of out[] in entries              */
    size_t       out_cnt;    /* actual entries written (set by thread)    */
} SMExtParJob;

static void *sm_ext_par_merge_thread(void *arg) {
    SMExtParJob *j   = (SMExtParJob *)arg;
    int          K   = j->K;
    SMExtStream *S   = j->streams;
    size_t  dyn_buf  = j->dyn_buf;
    SMEntry    *out  = j->out;
    size_t   out_cap = j->out_cap;

    /* Build initial min-heap over non-empty sub-streams. */
    SMHEntry *h = (SMHEntry *)malloc((size_t)K * sizeof(SMHEntry));
    int hs = 0;
    for (int i = 0; i < K; i++)
        if (S[i].buf_len > 0) { h[hs].key = S[i].buf[0].key; h[hs].s = i; hs++; }
    for (int i = hs/2-1; i >= 0; i--) sm_heap_sift(h, hs, i);

    size_t op = 0;

/* Helper macros for refilling a stream inside the heap loop. */
#define EXT_REFILL(sidx_)                                                   \
    do {                                                                     \
        SMExtStream *_s = &S[sidx_];                                         \
        if (_s->run_pos < _s->run_len) {                                     \
            size_t _rem = _s->run_len - _s->run_pos;                         \
            size_t _n   = _rem < dyn_buf ? _rem : dyn_buf;                   \
            size_t _off = _s->run_off + _s->run_pos * sizeof(SMEntry);       \
            if (_s->map) memcpy(_s->buf,(const char*)_s->map+_off,           \
                                _n*sizeof(SMEntry));                          \
            else pread(_s->fd, _s->buf, _n*sizeof(SMEntry), (off_t)_off);    \
            _s->buf_pos = 0; _s->buf_len = _n; _s->run_pos += _n;           \
            h[0].key = _s->buf[0].key;                                       \
            sm_heap_sift(h, hs, 0);                                          \
        } else {                                                              \
            h[0] = h[--hs];                                                  \
            if (hs > 0) sm_heap_sift(h, hs, 0);                              \
        }                                                                    \
    } while(0)

    while (hs > 0) {
        int     s_idx = h[0].s;
        SMEntry cur   = S[s_idx].buf[S[s_idx].buf_pos++];

        if (S[s_idx].buf_pos < S[s_idx].buf_len) {
            h[0].key = S[s_idx].buf[S[s_idx].buf_pos].key;
            sm_heap_sift(h, hs, 0);
        } else { EXT_REFILL(s_idx); }

        /* Dedup: accumulate all streams sharing the same key. */
        while (hs > 0 && h[0].key == cur.key) {
            int s2 = h[0].s;
            cur.val += S[s2].buf[S[s2].buf_pos++].val;
            if (S[s2].buf_pos < S[s2].buf_len) {
                h[0].key = S[s2].buf[S[s2].buf_pos].key;
                sm_heap_sift(h, hs, 0);
            } else { EXT_REFILL(s2); }
        }
#undef EXT_REFILL

        /* Write output — out_cap is an upper bound; no realloc needed. */
        if (op < out_cap) out[op] = cur;
        op++;
    }

    j->out_cnt = op;
    free(h);
    return NULL;
}

/* ── sm_global_ext_merge_par ─────────────────────────────────────────────
   Parallel K-way merge of all P workers' ext runs into nxt.

   Algorithm (replaces single-threaded sm_global_ext_merge):
     1. Sample SM_SAMPLES_PER_STREAM hash values from each of the K streams
        via random access into their mmap regions.
     2. Sort samples → pick T-1 splitters that divide the hash space evenly.
     3. Binary-search each stream to find per-splitter entry boundaries.
     4. Pre-allocate nxt->data using per-range entry counts as upper bounds.
     5. Launch T merge threads; each thread runs a K-way heap merge over its
        hash-range sub-streams, writing into a disjoint output slice.
     6. Compact: memmove to close dedup gaps, shrink allocation.

   Each thread reads its sub-streams via the existing buffered refill logic,
   so memory access remains largely sequential per thread.                   */
static size_t sm_global_ext_merge_par(SMWorkerState *WS, int P,
                                      SMTab *nxt, size_t raw_total,
                                      double *t_merge_ms_out) {
    /* Count total streams. */
    int total_streams = 0;
    for (int i = 0; i < P; i++) total_streams += WS[i].n_runs;

    if (total_streams == 0) {
        smtab_ensure(nxt, 1);
        nxt->cnt = 0;
        *t_merge_ms_out = 0;
        return 0;
    }

    /* Fall back to serial if too few streams to benefit from parallelism. */
    int T = sm_cfg.nthreads;           /* merge threads */
    if (T > total_streams) T = total_streams;
    if (T < 2) {
        /* Let the serial function handle it — call it via a compatible shim. */
        return sm_global_ext_merge(WS, P, nxt, raw_total, t_merge_ms_out);
    }

    /* Dynamic per-stream buffer: share SLC across all stream buffers. */
    size_t dyn_buf = (size_t)sm_cfg.slc_bytes
                     / ((size_t)total_streams * sizeof(SMEntry));
    if (dyn_buf < (size_t)sm_cfg.ext_stream_buf) dyn_buf = (size_t)sm_cfg.ext_stream_buf;
    if (dyn_buf > (size_t)(1 << 20))             dyn_buf = (size_t)(1 << 20);

    /* Build flat array of (fd, map, run_off, run_len) descriptors. */
    typedef struct { int fd; void *map; size_t run_off; size_t run_len; } StreamDesc;
    StreamDesc *SD = (StreamDesc *)malloc((size_t)total_streams * sizeof(StreamDesc));
    SM_ALLOC_CHECK(SD, (size_t)total_streams * sizeof(StreamDesc));
    {
        int si = 0;
        for (int w = 0; w < P; w++) {
            size_t file_off = 0;
            for (int r = 0; r < WS[w].n_runs; r++) {
                SD[si].fd      = WS[w].run_fd;
                SD[si].map     = WS[w].run_map;
                SD[si].run_off = file_off;
                SD[si].run_len = WS[w].run_lens[r];
                file_off += WS[w].run_lens[r] * sizeof(SMEntry);
                si++;
            }
        }
    }

    /* ── Phase 1: sample hash values from all streams ─────────────────── */
    int sps  = SM_SAMPLES_PER_STREAM;   /* samples per stream */
    int ns   = 0;
    u64 *samples = (u64 *)malloc((size_t)total_streams * sps * sizeof(u64));
    SM_ALLOC_CHECK(samples, (size_t)total_streams * sps * sizeof(u64));
    for (int k = 0; k < total_streams; k++) {
        size_t len = SD[k].run_len;
        if (len == 0) continue;
        for (int s = 0; s < sps; s++) {
            size_t idx = (size_t)s * (len > 1 ? len - 1 : 0) / (sps > 1 ? sps - 1 : 1);
            SMEntry e;
            size_t off = SD[k].run_off + idx * sizeof(SMEntry);
            if (SD[k].map) memcpy(&e, (const char*)SD[k].map + off, sizeof(SMEntry));
            else           pread(SD[k].fd, &e, sizeof(SMEntry), (off_t)off);
            samples[ns++] = eh_hash(e.key);
        }
    }
    /* Insertion sort (tiny: total_streams × sps ≤ 64×64 = 4096 values). */
    for (int i = 1; i < ns; i++) {
        u64 v = samples[i]; int j = i - 1;
        while (j >= 0 && samples[j] > v) { samples[j+1] = samples[j]; j--; }
        samples[j+1] = v;
    }
    /* Pick T-1 splitters at evenly spaced quantiles. */
    u64 *splitters = (u64 *)malloc((size_t)(T - 1) * sizeof(u64));
    SM_ALLOC_CHECK(splitters, (size_t)(T - 1) * sizeof(u64));
    for (int j = 0; j < T - 1; j++) {
        int idx = (int)((size_t)(j + 1) * ns / T);
        if (idx >= ns) idx = ns - 1;
        splitters[j] = samples[idx];
    }
    free(samples);
    /* Ensure strictly increasing. */
    for (int j = 1; j < T - 1; j++)
        if (splitters[j] <= splitters[j-1]) splitters[j] = splitters[j-1] + 1;

    /* ── Phase 2: binary-search boundaries per (stream, splitter) ─────── */
    /* bounds[k*(T+1) + t] = first entry index in stream k where hash >= splitter[t-1]
       (with splitter[-1] = 0 → bounds[k][0] = 0, splitter[T-1] = MAX → bounds[k][T] = len) */
    size_t *bounds = (size_t *)malloc((size_t)total_streams * (T + 1) * sizeof(size_t));
    SM_ALLOC_CHECK(bounds, (size_t)total_streams * (T + 1) * sizeof(size_t));
#define BND(k,t) bounds[(size_t)(k)*(T+1)+(t)]
    for (int k = 0; k < total_streams; k++) {
        BND(k,0) = 0;
        for (int t = 0; t < T - 1; t++)
            BND(k, t+1) = sm_ext_stream_lower_bound(
                SD[k].fd, SD[k].map, SD[k].run_off, SD[k].run_len, splitters[t]);
        BND(k, T) = SD[k].run_len;
    }
    free(splitters);

    /* ── Phase 3: compute per-range upper-bound entry counts & offsets ── */
    size_t *range_cap = (size_t *)malloc((size_t)T * sizeof(size_t));
    size_t *offsets   = (size_t *)malloc((size_t)(T + 1) * sizeof(size_t));
    SM_ALLOC_CHECK(range_cap, (size_t)T * sizeof(size_t));
    SM_ALLOC_CHECK(offsets,   (size_t)(T + 1) * sizeof(size_t));
    offsets[0] = 0;
    for (int t = 0; t < T; t++) {
        size_t cap = 0;
        for (int k = 0; k < total_streams; k++)
            cap += BND(k, t+1) - BND(k, t);
        range_cap[t] = cap;
        offsets[t+1] = offsets[t] + cap;
    }
    size_t total_upper = offsets[T];

    /* Allocate output (upper bound before dedup). */
    if (total_upper == 0) total_upper = 1;
    smtab_ensure(nxt, total_upper);

    /* ── Phase 4: build per-thread stream arrays and launch threads ───── */
    SMExtParJob *jobs  = (SMExtParJob *)malloc((size_t)T * sizeof(SMExtParJob));
    pthread_t   *thds  = (pthread_t   *)malloc((size_t)T * sizeof(pthread_t));
    SM_ALLOC_CHECK(jobs, (size_t)T * sizeof(SMExtParJob));
    SM_ALLOC_CHECK(thds, (size_t)T * sizeof(pthread_t));

    /* Each thread gets its own array of K SMExtStream descriptors. */
    SMExtStream *all_streams = (SMExtStream *)malloc(
        (size_t)T * total_streams * sizeof(SMExtStream));
    SM_ALLOC_CHECK(all_streams, (size_t)T * total_streams * sizeof(SMExtStream));

    double t0 = now_ms();

    for (int t = 0; t < T; t++) {
        SMExtStream *ts = &all_streams[(size_t)t * total_streams];
        for (int k = 0; k < total_streams; k++) {
            size_t lo  = BND(k, t);
            size_t hi  = BND(k, t+1);
            size_t len = hi - lo;
            ts[k].fd      = SD[k].fd;
            ts[k].map     = SD[k].map;
            ts[k].run_off = SD[k].run_off + lo * sizeof(SMEntry);
            ts[k].run_len = len;
            ts[k].run_pos = 0;
            ts[k].buf_pos = 0;
            ts[k].buf_len = 0;
            ts[k].buf     = (SMEntry *)malloc(dyn_buf * sizeof(SMEntry));
            SM_ALLOC_CHECK(ts[k].buf, dyn_buf * sizeof(SMEntry));
            /* Pre-fill first buffer block. */
            if (len > 0) {
                size_t n   = len < dyn_buf ? len : dyn_buf;
                size_t off = ts[k].run_off;
                if (ts[k].map) memcpy(ts[k].buf,(const char*)ts[k].map+off,n*sizeof(SMEntry));
                else           pread(ts[k].fd, ts[k].buf, n*sizeof(SMEntry), (off_t)off);
                ts[k].buf_len = n;
                ts[k].run_pos = n;
            }
        }
        jobs[t].streams  = ts;
        jobs[t].K        = total_streams;
        jobs[t].dyn_buf  = dyn_buf;
        jobs[t].out      = nxt->data + offsets[t];
        jobs[t].out_cap  = range_cap[t];
        jobs[t].out_cnt  = 0;
        pthread_create(&thds[t], NULL, sm_ext_par_merge_thread, &jobs[t]);
    }
    for (int t = 0; t < T; t++) pthread_join(thds[t], NULL);

    /* ── Phase 5: compact dedup gaps ─────────────────────────────────── */
    size_t actual = 0;
    for (int t = 0; t < T; t++) {
        size_t cnt = jobs[t].out_cnt;
        if (cnt > 0 && offsets[t] != actual)
            memmove(nxt->data + actual, nxt->data + offsets[t], cnt * sizeof(SMEntry));
        actual += cnt;
    }
    nxt->cnt = actual;

    /* Shrink allocation to actual size. */
    if (actual < total_upper) {
        SMEntry *p = (SMEntry *)sm_realloc(nxt->data, (actual > 0 ? actual : 1) * sizeof(SMEntry));
        if (p) { nxt->data = p; nxt->cap = (actual > 0 ? actual : 1); }
    }

    *t_merge_ms_out = now_ms() - t0;

    /* Cleanup. */
    for (int t = 0; t < T; t++)
        for (int k = 0; k < total_streams; k++)
            free(all_streams[(size_t)t * total_streams + k].buf);
    free(all_streams); free(jobs); free(thds);
    free(bounds); free(range_cap); free(offsets); free(SD);
#undef BND
    return actual;
}

/* ── sm_global_ext_merge ────────────────────────────────────────────────────
   Single-pass K-way merge directly from all P workers' ext (SSD) runs into
   nxt, bypassing the intermediate W[i].out RAM arrays entirely.

   This is the standard external-sort merge pass, generalised to cover all
   P workers' runs in one heap operation.

   Memory profile compared to the old two-phase approach
   (sm_merge_runs → W[i].out → pairwise/parallel merge → nxt):

     Old: curr + W_all + T_pair + OS    e.g. n=61 step 39 → 60 GB → swap
     New: curr + nxt  + stream_bufs     e.g. n=61 step 39 → 29 GB → no swap

   where stream_bufs = total_streams × sm_cfg.ext_stream_buf × 24B.
   For step 39: 258 streams × 2978 entries (sized to fill SLC) = 141 MB.

   Per-stream buffer sizing (the Aiken et al. capacity-boundary principle):
     Optimal buf ≈ SLC_bytes / total_streams / sizeof(SMEntry).
   We compute this per call so it adapts to the actual run count.
   Clamped below at sm_cfg.ext_stream_buf (the compile-time lower bound) and
   above at 1 MB (to avoid excessive malloc overhead for very few streams). */

/* SLC size hint — adjust for your machine.  The tuner (tune_params.py)
   reports the correct value.  On M3 Pro: 12582912 (12 MB).
   This constant is used only for the dynamic buffer-size formula.          */

static size_t sm_global_ext_merge(SMWorkerState *WS, int P,
                                   SMTab *nxt, size_t raw_total,
                                   double *t_merge_ms_out) {
    /* Count total streams and build a flat stream descriptor array. */
    int total_streams = 0;
    for (int i = 0; i < P; i++) total_streams += WS[i].n_runs;

    if (total_streams == 0) {
        smtab_ensure(nxt, 1);
        nxt->cnt = 0;
        *t_merge_ms_out = 0;
        return 0;
    }

    /* Dynamic per-stream buffer: fill SLC with all stream buffers.
       Clamped to [sm_cfg.ext_stream_buf, 1M] entries.                          */
    size_t dyn_buf = (size_t)sm_cfg.slc_bytes / ((size_t)total_streams * sizeof(SMEntry));
    if (dyn_buf < (size_t)sm_cfg.ext_stream_buf) dyn_buf = (size_t)sm_cfg.ext_stream_buf;
    if (dyn_buf > (size_t)(1 << 20))         dyn_buf = (size_t)(1 << 20);

    /* Allocate nxt with a RAM-safe initial capacity and grow ×1.5 as needed.
       We cannot use raw_total: at n=61 step 39, raw_total=2B × 24B = 48GB
       which exceeds the 36GB M3 Pro.  curr->data has already been freed by
       the caller (sm_fused_sweep) so the available RAM is ~32GB.
       Starting at 64M entries and growing ×1.5 reaches 677M in 6 reallocs;
       each realloc peaks at old×24 + new×24 < 32GB at all steps.          */
    {
        size_t init_cap = (size_t)64 << 20;   /* 64M entries = 1.5 GB    */
        if (init_cap < (size_t)total_streams * 2)
            init_cap = (size_t)total_streams * 2;
        smtab_ensure(nxt, init_cap);
    }

    /* Build one SMExtStream per run across all workers.                     */
    SMExtStream *S = (SMExtStream*)malloc((size_t)total_streams * sizeof(SMExtStream));
    SMHEntry    *h = (SMHEntry*)  malloc((size_t)total_streams * sizeof(SMHEntry));
    int si = 0;
    for (int w = 0; w < P; w++) {
        size_t file_off = 0;
        for (int r = 0; r < WS[w].n_runs; r++) {
            S[si] = (SMExtStream){
                .fd      = WS[w].run_fd,
                .map     = WS[w].run_map,
                .run_off = file_off,
                .run_len = WS[w].run_lens[r],
                .run_pos = 0,
                .buf_len = 0,
                .buf_pos = 0,
                .buf     = (SMEntry*)malloc(dyn_buf * sizeof(SMEntry)),
            };
            SM_ALLOC_CHECK(S[si].buf, dyn_buf * sizeof(SMEntry));
            /* Override the global SM_EXT_STREAM_BUF with the dynamic value. */
            S[si].buf_len = 0;  /* will be filled by refill below */
            file_off += WS[w].run_lens[r] * sizeof(SMEntry);
            /* Manual refill using dyn_buf instead of SM_EXT_STREAM_BUF.    */
            {
                SMExtStream *s = &S[si];
                size_t rem = s->run_len - s->run_pos;
                size_t n   = rem < dyn_buf ? rem : dyn_buf;
                size_t off = s->run_off + s->run_pos * sizeof(SMEntry);
                if (s->map)
                    memcpy(s->buf, (const char*)s->map + off, n * sizeof(SMEntry));
                else
                    pread(s->fd, s->buf, n * sizeof(SMEntry), (off_t)off);
                s->buf_pos = 0; s->buf_len = (size_t)n; s->run_pos += n;
            }
            si++;
        }
    }

    /* Build initial min-heap.                                               */
    int hs = 0;
    for (int i = 0; i < total_streams; i++)
        if (S[i].buf_len > 0) { h[hs].key = S[i].buf[0].key; h[hs].s = i; hs++; }
    for (int i = hs/2-1; i >= 0; i--) sm_heap_sift(h, hs, i);

    double t0 = now_ms();
    size_t op  = 0;

    /* K-way merge with dedup — same logic as the single-worker ext merge.
       Inline refill uses dyn_buf so SLC-fit is respected.                  */
    while (hs > 0) {
        int s_idx = h[0].s;
        SMEntry cur = S[s_idx].buf[S[s_idx].buf_pos++];

        /* Advance stream s_idx in the heap. */
        if (S[s_idx].buf_pos < S[s_idx].buf_len) {
            h[0].key = S[s_idx].buf[S[s_idx].buf_pos].key;
            sm_heap_sift(h, hs, 0);
        } else {
            /* Refill with dyn_buf. */
            SMExtStream *s = &S[s_idx];
            if (s->run_pos < s->run_len) {
                size_t rem = s->run_len - s->run_pos;
                size_t n   = rem < dyn_buf ? rem : dyn_buf;
                size_t off = s->run_off + s->run_pos * sizeof(SMEntry);
                if (s->map) memcpy(s->buf,(const char*)s->map+off,n*sizeof(SMEntry));
                else pread(s->fd, s->buf, n*sizeof(SMEntry), (off_t)off);
                s->buf_pos = 0; s->buf_len = n; s->run_pos += n;
                h[0].key = s->buf[0].key;
                sm_heap_sift(h, hs, 0);
            } else {
                h[0] = h[--hs];
                if (hs > 0) sm_heap_sift(h, hs, 0);
            }
        }

        /* Dedup: accumulate from all streams with the same key.             */
        while (hs > 0 && h[0].key == cur.key) {
            int s2 = h[0].s;
            cur.val += S[s2].buf[S[s2].buf_pos++].val;

            if (S[s2].buf_pos < S[s2].buf_len) {
                h[0].key = S[s2].buf[S[s2].buf_pos].key;
                sm_heap_sift(h, hs, 0);
            } else {
                SMExtStream *s = &S[s2];
                if (s->run_pos < s->run_len) {
                    size_t rem = s->run_len - s->run_pos;
                    size_t n   = rem < dyn_buf ? rem : dyn_buf;
                    size_t off = s->run_off + s->run_pos * sizeof(SMEntry);
                    if (s->map) memcpy(s->buf,(const char*)s->map+off,n*sizeof(SMEntry));
                    else pread(s->fd, s->buf, n*sizeof(SMEntry), (off_t)off);
                    s->buf_pos = 0; s->buf_len = n; s->run_pos += n;
                    h[0].key = s->buf[0].key;
                    sm_heap_sift(h, hs, 0);
                } else {
                    h[0] = h[--hs];
                    if (hs > 0) sm_heap_sift(h, hs, 0);
                }
            }
        }
        /* Grow nxt ×1.5 if needed: peak = old×24 + new×24 + OS < 36GB.   */
        if (op >= nxt->cap) {
            size_t new_cap = nxt->cap + (nxt->cap >> 1); /* ×1.5          */
            if (new_cap < nxt->cap + 1) new_cap = nxt->cap + 1;
            SMEntry *p = (SMEntry *)sm_realloc(nxt->data, new_cap * sizeof(SMEntry));
            if (!p) SM_OOM(new_cap * sizeof(SMEntry));
            nxt->data = p; nxt->cap = new_cap;
        }
        nxt->data[op++] = cur;
    }
    /* Shrink to exact size to free unused capacity. */
    if (op < nxt->cap) {
        SMEntry *p = (SMEntry *)sm_realloc(nxt->data, op * sizeof(SMEntry));
        if (p || op == 0) { nxt->data = p; nxt->cap = op; }
        /* If realloc-shrink fails, keep the oversized buffer — no data loss. */
    }

    *t_merge_ms_out = now_ms() - t0;
    nxt->cnt = op;

    for (int i = 0; i < total_streams; i++) free(S[i].buf);
    free(S); free(h);
    return op;
}

/* ── SM fused sweep ─────────────────────────────────────────────────── */
/* Predict whether run data will exceed available RAM.
   Uses full CAP as the per-run size estimate (conservative).  The old
   CAP/2 estimate assumed 2× intra-run dedup, which holds for low-nb
   steps but badly underestimates for nb=3/4 steps where keys are nearly
   unique (bpt >> state count) and runs are nearly full-sized.  Using CAP
   triggers ext mode earlier on genuinely large steps, trading time for
   memory safety.  The threshold only fires when run_bytes > avail, so
   small steps (low si or low nb) stay in RAM mode as before.            */
static int sm_should_use_ext(size_t si, int nb, size_t ram_bytes) {
    size_t P   = sm_cfg.nthreads;
    size_t CAP = sm_cfg.worker_cap;
    size_t chunk   = (si + P - 1) / P;
    size_t bpt     = chunk * ((size_t)1 << nb);
    size_t n_runs  = bpt > CAP ? (bpt + CAP - 1) / CAP : 1;
    size_t run_bytes  = n_runs * CAP * P * sizeof(SMEntry);  /* conservative: full CAP per run */
    size_t os_head    = (size_t)4 << 30;
    size_t curr_bytes = si * sizeof(SMEntry);
    size_t bufs_bytes = P * 2 * CAP * sizeof(SMEntry);
    if (ram_bytes <= os_head + curr_bytes + bufs_bytes) return 1;
    size_t avail = ram_bytes - os_head - curr_bytes - bufs_bytes;
    return run_bytes > avail;
}

static void sm_fused_sweep(SMTab *curr, SMTab *nxt,
                            int fs, int v_idx,
                            const int *widxs, int n_back,
                            const int *elim_idxs_desc, int n_elim,
                            int step, int n,
                            u128 *total,
                            size_t ram_bytes,
                            int instrument) {
    int P = sm_cfg.nthreads;
    size_t ni    = curr->cnt;
    size_t chunk = (ni + P - 1) / P;
    int ext_runs = sm_should_use_ext(ni, n_back, ram_bytes);

    /* Capacity for ext_runs backing store per worker.
       Must cover the true worst-case bytes written: all bpt entries before
       any dedup.  Using n_runs_per_w × CAP underestimates when the actual
       n_runs exceeds the formula (which happens when CAP is small relative
       to bpt), causing sm_ext_write to overflow the MAP_ANON region and
       subsequent reads to return zeros → silent data loss and wrong answers.
       MAP_ANON allocates physical pages lazily, so this large virtual
       reservation costs nothing until pages are actually written.         */
    size_t bpt_per_worker  = chunk * ((size_t)1 << n_back);
    size_t ext_cap_bytes   = bpt_per_worker * sizeof(SMEntry);

    SMWorker      *W  = (SMWorker*)calloc(P, sizeof(SMWorker));
    SMWorkerState *WS = (SMWorkerState*)calloc(P, sizeof(SMWorkerState));
    pthread_t     *T  = (pthread_t*)malloc(P * sizeof(pthread_t));
    pthread_mutex_t mu;
    pthread_mutex_init(&mu, NULL);

    for (int i = 0; i < P; i++) {
        size_t lo = (size_t)i * chunk;
        size_t hi = lo + chunk < ni ? lo + chunk : ni;
        WS[i].buf      = (SMEntry*)sm_alloc(sm_cfg.worker_cap * sizeof(SMEntry));
        WS[i].tmp      = (SMEntry*)sm_alloc(sm_cfg.worker_cap * sizeof(SMEntry));
        WS[i].buf_len    = 0;
        WS[i].ext_runs   = ext_runs;
        WS[i].n_runs     = 0;
        WS[i].runs_cap   = 0;
        WS[i].runs       = NULL;
        WS[i].run_fd     = -1;
        WS[i].run_map    = NULL;
        WS[i].run_capacity = 0;
        WS[i].run_fd_off = 0;
        WS[i].run_lens   = NULL;
        if (ext_runs) {
            WS[i].run_capacity = ext_cap_bytes;
            WS[i].run_fd = sm_ext_open(ext_cap_bytes, &WS[i].run_map);
        }
        W[i] = (SMWorker){
            .in=curr->data, .lo=lo, .hi=hi,
            .fs=fs, .v_idx=v_idx,
            .widxs=widxs, .n_back=n_back,
            .elim_idxs_desc=elim_idxs_desc, .n_elim=n_elim,
            .step=step, .n=n,
            .total=total, .total_mu=&mu,
            .ws=&WS[i]
        };
        pthread_create(&T[i], NULL, sm_worker_runs, &W[i]);
    }
    for (int i = 0; i < P; i++) pthread_join(T[i], NULL);
    pthread_mutex_destroy(&mu);

    /* Sum raw output counts (before any dedup) from each worker. */
    size_t raw_out_total = 0;
    for (int i = 0; i < P; i++) raw_out_total += W[i].raw_out;

    /* Compute raw_total for pre-allocation and instrumentation. */
    size_t raw_total = 0;
    size_t after_run_dedup_total = 0;  /* filled below per path */

    /* ── Merge strategy ─────────────────────────────────────────────────
       ext_runs path (large steps): sm_global_ext_merge
         All P workers' runs are already on SSD.  Merge directly from SSD
         into nxt in a single K-way heap pass; no W[i].out RAM arrays.
         Peak RAM: curr + nxt + stream_bufs (≈100-200 MB).
         Contrast with old two-phase (sm_merge_runs → pairwise):
           Peak RAM: curr + W_all + T_pair → 60 GB at n=61 step 39.

       RAM path (small/medium steps): sm_merge_runs (per-worker) then
         parallel or pairwise merge into nxt, as before.                 */
    double t_par0, t_par_merge, t_merge_runs_total = 0.0;
    int use_pairwise = 0;
    const char *merge_path;

    if (ext_runs) {
        /* Global ext merge: SSD → nxt directly.
           Flush any residual buf to ext first (workers already did this
           in sm_worker_runs, but guard for safety).                      */
        for (int i = 0; i < P; i++) {
            if (WS[i].buf_len > 0) sm_flush_run(&WS[i]);
            for (int r = 0; r < WS[i].n_runs; r++)
                raw_total += WS[i].run_lens[r];
        }
        after_run_dedup_total = raw_total;  /* ext: run_lens are already intra-run deduped */
        /* Free curr->data now: workers have finished reading it and
           global_ext_merge reads from WS[i] ext files, not curr.
           Freeing curr (up to 15 GB at n=61 step 39) ensures the
           ×1.5-growth reallocs in sm_global_ext_merge stay under 36 GB.
           curr->data = NULL makes the outer cleanup free() a no-op.       */
        sm_free(curr->data); curr->data = NULL; curr->cap = 0;

        /* Free worker scratch buffers before the merge: they held pre-flush
           data and are no longer needed now that all runs are on SSD.
           This saves P×2×sm_cfg.worker_cap×24B (4.6–9.2GB) during the merge,
           making the peak curr+nxt+stream_bufs+OS independent of CAP.    */
        for (int i = 0; i < P; i++) {
            sm_free(WS[i].buf); WS[i].buf = NULL;
            sm_free(WS[i].tmp); WS[i].tmp = NULL;
        }
        merge_path = "global_ext_par";
        double t_gm = 0;
        t_par0 = now_ms();
        sm_global_ext_merge_par(WS, P, nxt, raw_total, &t_gm);
        t_par_merge = now_ms() - t_par0;

        /* Release all ext backing stores now that merge is complete.    */
        for (int i = 0; i < P; i++) {
            sm_ext_close(WS[i].run_fd, WS[i].run_map, WS[i].run_capacity);
            WS[i].run_fd = -1; WS[i].run_map = NULL; WS[i].run_capacity = 0;
            free(WS[i].run_lens); WS[i].run_lens = NULL;
        }

        /* Shrink nxt to deduped size (same as RAM path).
           sm_global_ext_merge_par pre-allocates nxt at raw_total
           (e.g. 474 GB for step 60 raw_out=21189M) then fills only
           nxt->cnt entries (e.g. 179 GB after 2.6× dedup).  Without
           this shrink, curr->cap after the swap equals raw_total and
           the next sweep OOMs: curr(474) + workers(192) = 666 GB.    */
        if (nxt->cnt < nxt->cap) {
            if (nxt->cnt == 0) {
                /* All states eliminated: free data now so smtab_free
                   sees data=NULL and does not double-free.             */
                sm_free(nxt->data); nxt->data = NULL; nxt->cap = 0;
            } else {
                SMEntry *p = (SMEntry*)sm_realloc(nxt->data, nxt->cnt * sizeof(SMEntry));
                if (p) { nxt->data = p; nxt->cap = nxt->cnt; }
            }
        }

    } else {
        /* RAM path: per-worker merge_runs then final P-way merge.
           Free curr->data now: all workers have finished reading it.
           This saves curr_gb (up to 140 GB at n=75 step 44) during
           the sm_merge_runs + parallel-merge phase, which itself
           needs up to raw_out×24B of run data simultaneously.
           curr->data = NULL makes the outer free() a no-op.          */
        sm_free(curr->data); curr->data = NULL; curr->cap = 0;

        /* Stage 1: sum run lengths = entries after intra-run sort+dedup.
           Assigns to the outer after_run_dedup_total (not a new variable). */
        for (int i = 0; i < P; i++)
            for (int r = 0; r < WS[i].n_runs; r++)
                after_run_dedup_total += WS[i].runs[r].len;
        for (int i = 0; i < P; i++) {
            double tm0 = now_ms();
            W[i].out = sm_merge_runs(&WS[i], &W[i].out_len);
            W[i].out_size = W[i].out_len * sizeof(SMEntry);
            W[i].t_merge_ms = now_ms() - tm0;
            t_merge_runs_total += W[i].t_merge_ms;
            free(WS[i].runs); WS[i].runs = NULL;
        }
        for (int i = 0; i < P; i++) raw_total += W[i].out_len;

        /* Peak estimate for parallel: curr + W_all + nxt_raw + OS */
        size_t par_peak = (size_t)(ni + raw_total * 2) * sizeof(SMEntry)
                          + ((size_t)4 << 30);
        use_pairwise = (par_peak > ram_bytes);

        SMEntry **pw_arrs = (SMEntry**)malloc((size_t)P * sizeof(SMEntry*));
        size_t   *pw_lens = (size_t  *)malloc((size_t)P * sizeof(size_t));
        for (int i = 0; i < P; i++) {
            pw_arrs[i] = W[i].out; W[i].out = NULL;
            pw_lens[i] = W[i].out_len;
        }

        t_par0 = now_ms();
        if (use_pairwise) {
            merge_path = "pairwise";
            sm_pairwise_merge_into(pw_arrs, pw_lens, P, nxt);
        } else {
            merge_path = "parallel";
            smtab_ensure(nxt, raw_total + 1);
            SMStream *S = (SMStream*)malloc((size_t)P * sizeof(SMStream));
            for (int i = 0; i < P; i++)
                S[i] = (SMStream){ pw_arrs[i], 0, pw_lens[i] };
            nxt->cnt = sm_parallel_merge(S, P, P, nxt->data, nxt->cap);
            free(S);
            for (int i = 0; i < P; i++) {
                sm_free(pw_arrs[i]);
                pw_arrs[i] = NULL;
            }
        }
        t_par_merge = now_ms() - t_par0;
        free(pw_arrs);
        free(pw_lens);

        /* Shrink nxt->data to deduped size.
           smtab_ensure allocated at raw_total (e.g. 474 GB for ext step 60),
           but nxt->cnt is the deduped count (e.g. 179 GB).  Without this
           shrink, the next step's sweep sees curr->cap = raw_total and
           peak = curr_raw + workers → OOM.  Applies to BOTH ext and RAM paths.
           sm_realloc handles both platforms safely.                           */
        if (nxt->cnt < nxt->cap) {
            if (nxt->cnt == 0) {
                /* All states eliminated: free data now so smtab_free
                   sees data=NULL and does not double-free.             */
                sm_free(nxt->data); nxt->data = NULL; nxt->cap = 0;
            } else {
                SMEntry *p = (SMEntry*)sm_realloc(nxt->data, nxt->cnt * sizeof(SMEntry));
                if (p) { nxt->data = p; nxt->cap = nxt->cnt; }
            }
        }
    }  /* end RAM path */

    if (instrument) {
        /* Print sm_merge_bb buffer parameters for reference. */
        fprintf(stderr,
            "  [inst] merge_bb: SM_BB_BUF=%d SM_EXT_STREAM_BUF=%zu par_merge=%s\n",
            sm_cfg.bb_buf, sm_cfg.ext_stream_buf,
            merge_path);
        /* Per-worker breakdown */
        double t_compute_max = 0, t_flush_max = 0, t_merge_max = 0;
        for (int i = 0; i < P; i++) {
            if (W[i].t_compute_ms > t_compute_max) t_compute_max = W[i].t_compute_ms;
            if (W[i].t_flush_ms   > t_flush_max)   t_flush_max   = W[i].t_flush_ms;
            if (W[i].t_merge_ms   > t_merge_max)   t_merge_max   = W[i].t_merge_ms;
        }
        /* Imbalance: max worker time vs mean */
        double t_wall_compute = 0, t_wall_flush = 0;
        for (int i = 0; i < P; i++) {
            t_wall_compute += W[i].t_compute_ms;
            t_wall_flush   += W[i].t_flush_ms;
        }
        double dedup_ratio = raw_total > 0 ? (double)raw_total / nxt->cnt : 1.0;
        fprintf(stderr,
            "  [inst] step=%d nb=%d ext=%d n_runs=%d\n"
            "  [inst]   compute: max=%.0fms mean=%.0fms  "
            "flush: max=%.0fms mean=%.0fms\n"
            "  [inst]   merge_runs(seq): total=%.0fms max=%.0fms  "
            "par_merge: %.0fms\n"
            "  [inst]   raw_out=%zuM  deduped=%zuM  dedup=%.1fx\n"
            "  [inst]   dedup_stages: raw=%zuM  after_run=%zuM(%.2fx)  "
            "after_wmerge=%zuM(%.2fx)  final=%zuM(%.2fx)\n"
            "  [inst]   worker details (compute / flush / merge_runs / n_runs / out):\n",
            step, n_back, ext_runs,
            ext_runs ? (int)(raw_total / (sm_cfg.worker_cap/2)) : WS[0].n_runs,
            t_compute_max, t_wall_compute/P,
            t_flush_max,   t_wall_flush/P,
            t_merge_runs_total, t_merge_max,
            t_par_merge,
            raw_total/1000000, nxt->cnt/1000000, dedup_ratio,
            /* dedup_stages */
            raw_out_total/1000000,
            after_run_dedup_total/1000000,
            after_run_dedup_total>0 ? (double)raw_out_total/after_run_dedup_total : 1.0,
            /* after worker merge = raw_total for RAM path, same as after_run for ext */
            raw_total/1000000,
            raw_total>0 ? (double)after_run_dedup_total/raw_total : 1.0,
            nxt->cnt/1000000,
            raw_total>0 ? (double)raw_total/nxt->cnt : 1.0);
        for (int i = 0; i < P; i++) {
            fprintf(stderr,
                "  [inst]     w%-2d  %6.0fms / %6.0fms / %6.0fms / %2d / %zuK\n",
                i, W[i].t_compute_ms, W[i].t_flush_ms,
                W[i].t_merge_ms, WS[i].n_runs, W[i].out_len/1000);
        }
    }

    for (int i = 0; i < P; i++) {
        sm_free(W[i].out);
        sm_free(WS[i].buf);   /* may already be NULL if ext path freed early */
        sm_free(WS[i].tmp);   /* free(NULL) is a no-op per C99               */
        free(WS[i].runs);      /* NULL if already freed above */
        free(WS[i].run_lens);  /* NULL if already freed above */
        sm_ext_close(WS[i].run_fd, WS[i].run_map, WS[i].run_capacity);
    }
    free(W); free(WS); free(T);
}

/* ── Top-level SM entry point ───────────────────────────────────────── */
void count_ham_paths_sm(
    int         n,
    const int  *order,
    const int  *pos,
    const int  *last_s,
    const int  *adj_off,
    const int  *adj_dat,
    uint64_t   *res_lo,
    uint64_t   *res_hi,
    int         verbose,
    uint64_t    ram_bytes,
    int         instrument  /* 0=off, 1=per-step phase breakdown to stderr */
) {
    int  frontier[MAX_FS_FAST + 2];
    int *fidx = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int  fs   = 0;
    for (int i = 0; i <= n; i++) fidx[i] = -1;

    SMTab *curr = smtab_alloc(1024);
    SMTab *nxt  = smtab_alloc(1024);
    u128 total = (u128)0;

    curr->data[0].key = KEY_MARKER;
    curr->data[0].val = (u64)1;
    curr->cnt = 1;

    double t_start = now_ms(), t_prev = t_start;
    if (verbose)
        fprintf(stderr,
            "step  vertex  fs  n_back  e_bag  states_in  states_out"
            "  step_ms  cumul_ms  bag  [sm]\n");

    for (int step = 0; step < n; step++) {
        int v = order[step];
        if (fs > MAX_FS_FAST) { *res_lo = *res_hi = UINT64_MAX; goto cleanup; }

        /* A. Introduce: transform curr->data in-place.
           introduce(key, fs) ORs enc_v(0)<<(4*fs) into each key — it only
           touches bits at position fs and is independent per entry, so P
           threads can do N/P entries each with no synchronisation.
           In-place avoids allocating a separate nxt here, halving peak RAM:
             Old (copy): curr(si) + nxt(si) = 2×si simultaneously
             New (in-place): curr(si) only — nxt allocated later in sweep
           At n=75 step 57 (13.3B states = 303GB) this saves 303GB.      */
        double t_intro0 = now_ms();
        sm_introduce(curr, fs);
        double t_intro_ms = now_ms() - t_intro0;
        if (instrument)
            fprintf(stderr, "  [inst] introduce: si=%zuM  %.0fms\n",
                    curr->cnt / 1000000, t_intro_ms);
        /* curr->data now holds the introduced keys; pass directly to sweep.
           No swap needed — curr IS the introduced table.                  */
        frontier[fs] = v; fidx[v] = fs; fs++;

        /* nxt is still empty (data=NULL from previous step's free or initial
           state).  sm_fused_sweep will allocate it via smtab_ensure.      */

        /* B+C. Fused sweep */
        int v_idx = fs - 1;
        size_t states_in = curr->cnt;
        int n_back = 0, widxs[8];
        for (int ai = adj_off[v]; ai < adj_off[v+1]; ai++) {
            int w = adj_dat[ai];
            if (fidx[w] >= 0 && pos[w] < pos[v] && n_back < 8)
                widxs[n_back++] = fidx[w];
        }
        int elim_asc[MAX_FS_FAST+2], n_elim = 0;
        for (int ei = 0; ei < fs; ei++)
            if (last_s[frontier[ei]] <= step) elim_asc[n_elim++] = ei;
        int elim_desc[MAX_FS_FAST+2];
        for (int e = 0; e < n_elim; e++)
            elim_desc[e] = elim_asc[n_elim-1-e];

        /* resolve ram_bytes: use provided value or conservative 8 GB default */
        size_t ram = ram_bytes ? (size_t)ram_bytes : (size_t)8 << 30;

        sm_fused_sweep(curr, nxt, fs, v_idx, widxs, n_back,
                       elim_desc, n_elim, step, n, &total, ram, instrument);
        /* Free curr->data (the in-place-introduced table workers just read).
           The ext path already frees it inside sm_fused_sweep (sets to NULL);
           the RAM path does not, so this handles both: free(NULL) is a no-op. */
        sm_free(curr->data); curr->data = NULL; curr->cap = 0; curr->cnt = 0;
        { SMTab *t = curr; curr = nxt; nxt = t; }

        /* Update frontier */
        for (int e = n_elim-1; e >= 0; e--) {
            int ei = elim_asc[e];
            fidx[frontier[ei]] = -1;
            for (int k = ei; k < fs-1; k++) {
                frontier[k] = frontier[k+1]; fidx[frontier[k]] = k;
            }
            fs--;
        }

        if (verbose) {
            /* Count intra-frontier edges (both endpoints in frontier) */
            int e_bag = 0;
            for (int fi = 0; fi < fs; fi++) {
                int u = frontier[fi];
                for (int ai = adj_off[u]; ai < adj_off[u+1]; ai++) {
                    int w = adj_dat[ai];
                    if (fidx[w] >= 0 && fidx[w] > fi)  /* count each edge once */
                        e_bag++;
                }
            }
            /* Print frontier vertex list as {v1,v2,...} */
            char bag_buf[256];
            int  bpos = 0;
            bag_buf[bpos++] = '{';
            for (int fi = 0; fi < fs && bpos < 230; fi++) {
                if (fi > 0) bag_buf[bpos++] = ',';
                bpos += snprintf(bag_buf + bpos, sizeof(bag_buf) - bpos - 2,
                                 "%d", frontier[fi]);
            }
            bag_buf[bpos++] = '}';
            bag_buf[bpos]   = '\0';

            double t_now = now_ms();
            size_t ram = ram_bytes ? (size_t)ram_bytes : (size_t)8 << 30;
            const char *mode = sm_should_use_ext(states_in, n_back, ram)
                               ? "[ext]" : "[sm]";
            fprintf(stderr,
                "%4d  %6d  %2d  %6d  %5d  %9zu  %10zu  %8.1f  %8.1f  %-*s  %s\n",
                step, v, fs, n_back, e_bag,
                states_in, curr->cnt,
                t_now - t_prev, t_now - t_start,
                (int)(2 + fs * 4), bag_buf, mode);
            t_prev = t_now;
        }
    }

    *res_lo = (uint64_t) total;
    *res_hi = (uint64_t)(total >> 64);

cleanup:
    smtab_free(curr); smtab_free(nxt);
    free(fidx);
}
