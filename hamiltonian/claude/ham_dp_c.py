"""
ham_dp_c.py  (optimised v2)
---------------------------
C-accelerated frontier DP for counting Hamiltonian paths in G_n.

Key optimisations over v1
--------------------------
1. Packed uint64_t state keys (4 bits per frontier slot + 2 flag bits).
   Hash and compare become single-integer operations.

2. Memory-efficient hash tables: no used_list, keys+vals only (24 bytes/slot
   vs 32 before), 75% load factor (vs 50%), sequential memset for clear.
   v1 called malloc/calloc/free on every edge and every elimination
   (~200+ allocations for n=57).  v2 pre-allocates two tables and
   clears only occupied slots between transitions.

3. Faster hash: two-round 64-bit finaliser mix.

4. Faster canonicalisation: nibble loop on a register value.

State encoding (fs <= MAX_FS_FAST = 15)
----------------------------------------
  bits 4i..4i+3  (i=0..fs-1) : connectivity nibble for slot i
                                 nibble = value + 2
                                   1  <->  -1  (interior)
                                   2  <->   0  (fresh)
                                 k+2  <->   k  (label k, k=1..13)
  bit 62  : n_closed flag
  bit 63  : always 1 for valid states; 0 = empty hash slot

4 bits handles labels up to 13, covering frontier sizes up to 13
(4*13 + 2 flag bits = 54 <= 64).

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

/* Milliseconds since an arbitrary epoch (monotonic). */
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

typedef unsigned __int128 u128;
typedef uint64_t          u64;

#define KEY_MARKER  ((u64)1 << 63)
#define KEY_NC_BIT  ((u64)1 << 62)
/* Maximum frontier size for the packed uint64_t fast path.
   With 4 bits per slot and 2 reserved flag bits (KEY_MARKER, KEY_NC_BIT):
     4 * 15 + 2 = 62 bits  <=  64 bits  -- exactly fits.
     4 * 16 + 2 = 66 bits  >   64 bits  -- would overflow.
   So 15 is the hard upper limit for the packed encoding. */
#define MAX_FS_FAST 15

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

static inline int nc_get(u64 key)        { return (int)((key >> 62) & 1u); }
static inline u64 nc_set_1(u64 key)      { return key | KEY_NC_BIT; }

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

/* Append a fresh (value=0) slot at position fs. */
static inline u64 introduce(u64 key, int fs)
{
    return key | ((u64)enc_v(0) << (4*fs));
}

/* Remove slot u_idx; shift higher slots down. */
static inline u64 eliminate_slot(u64 key, int u_idx, int fs)
{
    u64 low  = key & (((u64)1 << (4*u_idx)) - 1u);
    u64 high = (key >> (4*(u_idx+1)))
               & (((u64)1 << (4*(fs - u_idx - 1))) - 1u);
    return (key & (KEY_MARKER | KEY_NC_BIT)) | low | (high << (4*u_idx));
}

/* ======================================================================
   Hash table: u64 key -> u128 value
   key==0 is the empty-slot sentinel; all valid keys have bit 63 set.
   ====================================================================== */
typedef struct {
    u64    *keys;
    u128   *vals;
    size_t  cap;
    size_t  cnt;
} HT;

static inline size_t ht_hash(u64 k, size_t cap)
{
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return (size_t)(k & (cap - 1));
}

static void *ht_calloc_huge(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
#ifdef MADV_HUGEPAGE
    if (p && n * sz >= (2u << 20))
        madvise(p, n * sz, MADV_HUGEPAGE);
#endif
    return p;
}

/* Set to 1 by ht_add_capped when the table hits the slot cap at ≥90%% load.
   The main DP loop checks this flag after every fused_sweep and aborts
   early if it is set, returning UINT64_MAX as a sentinel to Python.    */
static volatile int ht_overflow = 0;

static HT *ht_alloc(size_t cap)
{
    HT *t   = (HT *)malloc(sizeof(HT));
    t->cap  = cap;
    t->cnt  = 0;
    t->keys = (u64  *)ht_calloc_huge(cap, sizeof(u64));
    t->vals = (u128 *)ht_calloc_huge(cap, sizeof(u128));
    return t;
}

static void ht_free(HT *t)
{
    free(t->keys); free(t->vals); free(t);
}

/* Clear: zero keys and vals arrays.
   Sequential memset runs at memory-bandwidth speed (~50 GB/s).
   Both arrays must be zeroed: keys because 0 is the empty-slot sentinel,
   vals because ht_add accumulates with +=, so stale vals would corrupt counts. */
static void ht_clear(HT *t)
{
    memset(t->keys, 0, t->cap * sizeof(u64));
    memset(t->vals, 0, t->cap * sizeof(u128));
    t->cnt = 0;
}

/* Ensure cap == next_pow2(min_entries / LOAD_NUM * LOAD_DEN), then clear.
   Load factor target: 75% (cap = ceil(min_entries * 4/3) rounded to pow2).
   75% load vs old 50% load halves the wasted capacity, saving ~cap/2 slots
   = (8+16) * cap/2 bytes = 12 bytes/state in memory.
   Resize both UP and DOWN so post-peak steps don't keep giant tables. */
/* Load-factor numerator/denominator (runtime-settable).
   Default 4/3 → 75% load.  Pass --load-factor 85 to use 6/5 → 85%.
   Higher load = smaller tables (saves memory) but more probes per insert.
   Supported named values:
     75  →  4/3   (2.5 avg probes, default)
     80  →  5/4   (3.4 avg probes)
     85  →  6/5   (5.5 avg probes)
     90  →  9/8  (10.5 avg probes)                                       */
static int LOAD_NUM = 4;
static int LOAD_DEN = 3;

/* ht_ensure_clear: size and clear t to hold at least min_entries at LOAD_DEN/LOAD_NUM
   load, but never exceed max_slots (the memory budget for this table).
   If min_entries / (LOAD_DEN/LOAD_NUM) > max_slots, we accept a higher load factor
   rather than swap to disk — high load means more probes per insert but stays in RAM.
   max_slots == 0 means unlimited (original behaviour).                            */
static void ht_ensure_clear(HT *t, size_t min_entries, size_t max_slots)
{
    if (min_entries < 8) min_entries = 8;
    /* Ideal capacity at the target load factor. */
    size_t needed = 16;
    while (needed * LOAD_DEN < min_entries * LOAD_NUM) needed *= 2;

    /* Apply memory cap: never exceed max_slots. */
    if (max_slots > 0 && needed > max_slots) {
        /* Round max_slots DOWN to a power of 2. */
        size_t capped = 1;
        while (capped * 2 <= max_slots) capped *= 2;
        if (capped < 16) capped = 16;
        needed = capped;
    }

    if (t->cap == needed) {
        ht_clear(t);
        return;
    }
    free(t->keys); free(t->vals);
    t->cap  = needed;
    t->cnt  = 0;
    t->keys = (u64  *)ht_calloc_huge(needed, sizeof(u64));
    t->vals = (u128 *)ht_calloc_huge(needed, sizeof(u128));
}

/* ht_add_capped: like ht_add but honours a per-table slot cap.
   When the table would grow beyond max_slots, do NOT resize; accept a
   higher load factor.  Linear probing still works correctly (just slower)
   as long as at least one free slot exists — guaranteed while cnt < max_slots. */
static void ht_add_capped(HT *t, u64 key, u128 val, size_t max_slots)
{
    /* Grow at 75% load, but only up to max_slots. */
    if (t->cnt * LOAD_NUM >= t->cap * LOAD_DEN) {
        size_t nc = t->cap * 2;
        if (max_slots > 0 && nc > max_slots) {
            /* Cannot grow: clamp to max_slots (power of 2). */
            size_t capped = 1;
            while (capped * 2 <= max_slots) capped *= 2;
            nc = capped;
        }
        if (nc > t->cap) {           /* only resize if we can actually grow */
            u64   *ok   = t->keys;
            u128  *ov   = t->vals;
            size_t ocap = t->cap;
            t->cap  = nc;
            t->cnt  = 0;
            t->keys = (u64  *)calloc(nc, sizeof(u64));
            t->vals = (u128 *)calloc(nc, sizeof(u128));
            for (size_t i = 0; i < ocap; i++) {
                if (!ok[i]) continue;
                size_t idx = ht_hash(ok[i], nc);
                while (t->keys[idx]) idx = (idx + 1) & (nc - 1);
                t->keys[idx] = ok[i];
                t->vals[idx] = ov[i];
                t->cnt++;
            }
            free(ok); free(ov);
        } else if (max_slots > 0 && t->cnt * 10 >= t->cap * 9) {
            /* Resize was refused because we are already at the cap, and the
               table has now reached ≥90%% occupancy.  At this load factor
               linear probing degrades (avg >5 probes) and the risk of an
               infinite loop approaches.  Set the global flag so the DP loop
               can abort cleanly and report the problem to the caller.       */
            ht_overflow = 1;
        }
    }
    size_t idx = ht_hash(key, t->cap);
    while (t->keys[idx] && t->keys[idx] != key)
        idx = (idx + 1) & (t->cap - 1);
    if (!t->keys[idx]) {
        t->keys[idx] = key;
        t->cnt++;
    }
    t->vals[idx] += val;
}

/* Insert key->+val.  Grows at 75% load. */
static void ht_add(HT *t, u64 key, u128 val)
{
    /* Grow when cnt/cap > 3/4. */
    if (t->cnt * LOAD_NUM >= t->cap * LOAD_DEN) {
        size_t nc   = t->cap * 2;
        u64   *ok   = t->keys;
        u128  *ov   = t->vals;
        size_t ocap = t->cap;
        t->cap  = nc;
        t->cnt  = 0;
        t->keys = (u64  *)ht_calloc_huge(nc, sizeof(u64));
        t->vals = (u128 *)ht_calloc_huge(nc, sizeof(u128));
        /* Rehash all old entries. */
        for (size_t i = 0; i < ocap; i++) {
            if (!ok[i]) continue;
            size_t idx = ht_hash(ok[i], nc);
            while (t->keys[idx]) idx = (idx + 1) & (nc - 1);
            t->keys[idx] = ok[i];
            t->vals[idx] = ov[i];
            t->cnt++;
        }
        free(ok); free(ov);
    }
    size_t idx = ht_hash(key, t->cap);
    while (t->keys[idx] && t->keys[idx] != key)
        idx = (idx + 1) & (t->cap - 1);
    if (!t->keys[idx]) {
        t->keys[idx] = key;
        t->cnt++;
    }
    t->vals[idx] += val;
}

/* ======================================================================
   apply_single_edge: apply edge (v_idx, w_idx) to key nk.
   Returns the updated key (canonicalised), or 0 if the edge is invalid
   (would create a cycle, or a vertex is already interior).
   Sets *new_nc to 1 if this completes a half-external chain merge.
   ====================================================================== */
static inline u64 apply_edge(u64 nk, int fs,
                              int v_idx, int w_idx, int *new_nc)
{
    int8_t sv = slot_get(nk, v_idx);
    int8_t sw = slot_get(nk, w_idx);

    if (sv == -1 || sw == -1) return 0;   /* already interior */

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
    if (sv == sw) return 0;                /* cycle */

    /* Merge chains sv and sw. */
    int sv_c = label_count(nk, fs, sv);
    int sw_c = label_count(nk, fs, sw);
    nk = slot_set(nk, v_idx, -1);
    nk = slot_set(nk, w_idx, -1);
    nk = label_rename(nk, fs, sw, sv);

    if (sv_c == 1 && sw_c == 1) {
        /* Half-external merge: both chains close. */
        if (nc_get(nk) >= 1) return 0;    /* already have closed chain */
        for (int k = 0; k < fs; k++)
            if (slot_get(nk, k) != -1) return 0;  /* other open chains */
        *new_nc = 1;
        return canon(nc_set_1(nk), fs);
    }
    return canon(nk, fs);
}

/* ======================================================================
   apply_elim_seq: apply a sequence of n_elim eliminations to key nk
   (which has cur_fs slots).  Elimination indices are given in DESCENDING
   order so that removing a high slot does not shift the indices of lower
   pending slots.

   Returns the final canonical key, or 0 if the state is invalid (unvisited
   vertex, excess closed chains, or isolated chain at intermediate step).
   Final-step Hamiltonian path completions are accumulated into *total.
   ====================================================================== */
static inline u64 apply_elim_seq(u64 nk, int cur_fs,
                                  const int *elim_idxs_desc, int n_elim,
                                  int step, int n, u128 cnt, u128 *total)
{
    for (int e = 0; e < n_elim; e++) {
        int    u_idx = elim_idxs_desc[e];
        int8_t su    = slot_get(nk, u_idx);

        if (su == 0) return 0;   /* unvisited vertex: invalid path */

        u64 rk = eliminate_slot(nk, u_idx, cur_fs);
        cur_fs--;

        if (su == -1) {
            nk = canon(rk, cur_fs);
        } else {
            /* su = L: check if partner is still in the remaining slots. */
            int partner = 0;
            for (int k = 0; k < cur_fs; k++)
                if (slot_get(rk, k) == su) { partner = 1; break; }

            if (partner) {
                nk = canon(rk, cur_fs);
            } else {
                /* Both endpoints external → chain complete. */
                if (nc_get(nk) >= 1) return 0;   /* 2nd closed chain */
                for (int k = 0; k < cur_fs; k++)
                    if (slot_get(rk, k) != -1) return 0; /* open chains remain */

                /* Mark chain closed.  Remaining slots (if any) are all -1
                   and will be cleanly eliminated in subsequent iterations. */
                nk = nc_set_1(canon(rk, cur_fs));
                /* If this is the last elimination and the frontier is now
                   empty, count as a Hamiltonian path on the final step. */
                if (cur_fs == 0) {
                    if (step == n - 1) *total += cnt;
                    return 0;   /* do not add to nxt table */
                }
            }
        }
    }

    /* If nc == 1 and frontier is empty, count on final step. */
    if (cur_fs == 0) {
        if (step == n - 1 && nc_get(nk) == 1) *total += cnt;
        return 0;
    }
    return nk;
}

/* ======================================================================
   fused_sweep: single-pass sweep that applies all 2^n_back edge subsets
   AND all n_elim eliminations together for each source state.

   This avoids creating a large intermediate table between B and C.
   When n_back=3 at 4M states with 2 eliminations (final output ~1M):
     Old: intermediate ~7M → 16M-slot table → elimination scans 16M slots
     New: output ~1M → 2M-slot table, only one source scan of curr

   Elimination indices must be given in DESCENDING order.
   n_back == 0 is valid (pure elimination sweep).

   The expansion ratio for each step varies from 0.2× to 4×.
   Starting at 1.5× provides a good default; ht_add will grow the table
   if the output exceeds this.  ht_compact then right-sizes the result
   before it becomes curr, so the occasional resize does not cascade.

   Elimination indices must be given in DESCENDING order.
   n_back == 0 is valid (pure elimination sweep).
   ====================================================================== */
#define FUSED_MIN_STATES 200000

static void fused_sweep(HT *curr, HT *nxt,
                        int fs, int v_idx,
                        const int *widxs, int n_back,
                        const int *elim_idxs_desc, int n_elim,
                        int step, int n, u128 *total,
                        size_t slot_cap)
{
    int n_subsets = 1 << n_back;
    int final_fs  = fs - n_elim;

    /* Pre-allocate at 1.5× curr->cnt, respecting the slot cap.  ht_add_capped
       will grow if needed but never exceed slot_cap slots. */
    ht_ensure_clear(nxt, curr->cnt + curr->cnt / 2 + 16, slot_cap);

    for (size_t i = 0; i < curr->cap; i++) {
        if (!curr->keys[i]) continue;
        u64  base = curr->keys[i];
        u128 cnt  = curr->vals[i];

        for (int S = 0; S < n_subsets; S++) {
            u64 nk    = base;
            int valid = 1;

            /* Apply selected back-edges for subset S. */
            for (int j = 0; j < n_back && valid; j++) {
                if (!(S & (1 << j))) continue;
                int nc_inc = 0;
                u64 nk2 = apply_edge(nk, fs, v_idx, widxs[j], &nc_inc);
                if (!nk2) { valid = 0; break; }
                nk = nk2;
            }
            if (!valid) continue;

            /* Apply elimination sequence (returns 0 if invalid or counted). */
            if (n_elim > 0) {
                nk = apply_elim_seq(nk, fs, elim_idxs_desc, n_elim,
                                    step, n, cnt, total);
                if (!nk) continue;
            }

            ht_add_capped(nxt, nk, cnt, slot_cap);
            if (ht_overflow) return;    /* abort: table cap exceeded */
        }
    }
}

/* ======================================================================
   Checkpointing
   -----------------------------------------------------------------------
   Binary format (little-endian):
     4 bytes  magic   0xC8EC7A1E
     4 bytes  version 1
     4 bytes  n
     4 bytes  step    (last COMPLETED step, -1 means not started)
     4 bytes  fs      (frontier size after step)
     8 bytes  total_lo
     8 bytes  total_hi
     fs×4     frontier[]
     8 bytes  cnt     (number of occupied hash table entries)
     cnt×8    keys[]
     cnt×16   vals[]   (u128 as two u64, lo then hi)

   At 60M states: ~60M × 24 bytes = ~1.4 GB per checkpoint.
   ====================================================================== */
#define CKPT_MAGIC   0xC8EC7A1Eu
#define CKPT_VERSION 2u   /* v2 adds 8-byte ordering hash */

/* FNV-1a 64-bit hash of the vertex ordering array. */
static uint64_t order_hash(int n, const int *order)
{
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)order[i];
        h ^= (v & 0xff);        h *= 1099511628211ULL;
        h ^= ((v >>  8) & 0xff); h *= 1099511628211ULL;
        h ^= ((v >> 16) & 0xff); h *= 1099511628211ULL;
        h ^= ((v >> 24) & 0xff); h *= 1099511628211ULL;
    }
    return h;
}

static int ckpt_save(const char *path, int n, const int *order,
                     int step, int fs,
                     const int *frontier, u128 total, HT *curr)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    uint32_t magic   = CKPT_MAGIC;
    uint32_t version = CKPT_VERSION;
    uint32_t un      = (uint32_t)n;
    uint32_t ustep   = (uint32_t)step;
    uint32_t ufs     = (uint32_t)fs;
    uint64_t ord_h   = order_hash(n, order);  /* ordering fingerprint */
    uint64_t tot_lo  = (uint64_t)total;
    uint64_t tot_hi  = (uint64_t)(total >> 64);
    uint64_t cnt     = (uint64_t)curr->cnt;

    if (fwrite(&magic,   4, 1, f) != 1) goto fail;
    if (fwrite(&version, 4, 1, f) != 1) goto fail;
    if (fwrite(&un,      4, 1, f) != 1) goto fail;
    if (fwrite(&ustep,   4, 1, f) != 1) goto fail;
    if (fwrite(&ufs,     4, 1, f) != 1) goto fail;
    if (fwrite(&ord_h,   8, 1, f) != 1) goto fail;  /* v2: ordering hash */
    if (fwrite(&tot_lo,  8, 1, f) != 1) goto fail;
    if (fwrite(&tot_hi,  8, 1, f) != 1) goto fail;
    if (fs > 0 && fwrite(frontier, 4*(size_t)fs, 1, f) != 1) goto fail;
    if (fwrite(&cnt,     8, 1, f) != 1) goto fail;

    /* Write occupied entries only. */
    for (size_t i = 0; i < curr->cap; i++) {
        if (!curr->keys[i]) continue;
        uint64_t val_lo = (uint64_t) curr->vals[i];
        uint64_t val_hi = (uint64_t)(curr->vals[i] >> 64);
        if (fwrite(&curr->keys[i], 8, 1, f) != 1) goto fail;
        if (fwrite(&val_lo,        8, 1, f) != 1) goto fail;
        if (fwrite(&val_hi,        8, 1, f) != 1) goto fail;
    }
    fclose(f);
    return 1;
fail:
    fclose(f);
    return 0;
}

static int ckpt_load(const char *path, int n_expected,
                     const int *order,          /* current ordering to verify */
                     int *step_out, int *fs_out, int *frontier_out,
                     u128 *total_out, HT *curr,
                     size_t slot_cap)
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
    if (fread(&ord_h_saved, 8, 1, f) != 1) goto fail;   /* v2 */

    /* Reject checkpoint if the ordering changed. */
    if (ord_h_saved != order_hash(n_expected, order)) {
        fprintf(stderr,
            "# Checkpoint ordering mismatch — discarding stale checkpoint.\n");
        goto fail;
    }

    if (fread(&tot_lo,  8, 1, f) != 1) goto fail;
    if (fread(&tot_hi,  8, 1, f) != 1) goto fail;

    *step_out  = (int)ustep;
    *fs_out    = (int)ufs;
    *total_out = ((u128)tot_hi << 64) | (u128)tot_lo;

    if ((int)ufs > 0) {
        if (fread(frontier_out, 4*(size_t)ufs, 1, f) != 1) goto fail;
    }
    if (fread(&cnt, 8, 1, f) != 1) goto fail;

    /* Rebuild hash table. */
    ht_ensure_clear(curr, (size_t)cnt, slot_cap);
    for (uint64_t e = 0; e < cnt; e++) {
        uint64_t key, val_lo, val_hi;
        if (fread(&key,    8, 1, f) != 1) goto fail;
        if (fread(&val_lo, 8, 1, f) != 1) goto fail;
        if (fread(&val_hi, 8, 1, f) != 1) goto fail;
        u128 val = ((u128)val_hi << 64) | (u128)val_lo;
        ht_add_capped(curr, key, val, slot_cap);
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
    int        verbose,           /* 0 = silent; 1 = per-step stats to stderr */
    size_t     max_table_slots,   /* per-table slot cap (0 = unlimited)        */
    const char *checkpoint_path,  /* path for checkpoint file (NULL = disabled)*/
    double     checkpoint_secs,   /* save checkpoint every N seconds (0 = never)*/
    int        load_num,          /* load factor numerator   (e.g. 4 for 75%)  */
    int        load_den           /* load factor denominator (e.g. 3 for 75%)  */
) {
    /* Apply caller-specified load factor (e.g. 4/3 = 75%, 6/5 = 85%). */
    if (load_num > 0 && load_den > 0) { LOAD_NUM = load_num; LOAD_DEN = load_den; }
    size_t slot_cap = max_table_slots;
    int  frontier[MAX_FS_FAST + 2];
    int *fidx = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int  fs   = 0;
    for (int i = 0; i <= n; i++) fidx[i] = -1;

    HT *curr = ht_alloc(16);
    HT *nxt  = ht_alloc(16);

    u128 total = (u128)0;
    int start_step = 0;

    /* Try to load checkpoint if one exists. */
    if (checkpoint_path && checkpoint_path[0]) {
        int ckpt_step = -1, ckpt_fs = 0;
        if (ckpt_load(checkpoint_path, n, order, &ckpt_step, &ckpt_fs,
                      frontier, &total, curr, slot_cap)) {
            start_step = ckpt_step + 1;
            fs = ckpt_fs;
            /* Restore fidx from frontier. */
            for (int i = 0; i < fs; i++) fidx[frontier[i]] = i;
            if (verbose)
                fprintf(stderr, "# Resumed from checkpoint at step %d "
                        "(%zu states)\n", ckpt_step, curr->cnt);
        }
    }

    /* If starting fresh, insert initial state. */
    if (start_step == 0)
        ht_add(curr, KEY_MARKER, (u128)1);

    double t_start = now_ms();
    double t_prev  = t_start;
    double t_last_ckpt = t_start;

    if (verbose)
        fprintf(stderr,
            "step  vertex  fs  n_back  states_in  states_out"
            "  step_ms  cumul_ms\n");

    for (int step = start_step; step < n; step++) {
        int v = order[step];

        if (fs >= MAX_FS_FAST) {
            *res_lo = *res_hi = UINT64_MAX;  /* signal error */
            goto cleanup;
        }

        /* ---- A. Introduce v ------------------------------------------ */
        ht_ensure_clear(nxt, curr->cnt + 4, slot_cap);
        for (size_t i = 0; i < curr->cap; i++) {
            if (!curr->keys[i]) continue;
            ht_add(nxt, introduce(curr->keys[i], fs), curr->vals[i]);
        }
        { HT *tmp = curr; curr = nxt; nxt = tmp; }
        frontier[fs] = v;
        fidx[v]      = fs;
        fs++;

        /* ---- B + C. Edge decisions and elimination ----------------------
           Precompute: back-edge neighbour indices (widxs) and the indices
           of vertices to be eliminated at this step (elim_idxs_desc,
           sorted DESCENDING so that removing a high slot does not shift
           the indices of lower pending slots in apply_elim_seq).

           When n_elim >= 1 and the table is large, use fused_sweep:
           apply all 2^n_back edge subsets AND all eliminations in ONE
           source-table scan, writing directly to the final (post-elim)
           output.  This eliminates the large intermediate table that
           previously sat between B and C, which was the dominant cause
           of slow elimination scans (steps 34 and 42 in the n=57 profile).

           When n_elim == 0: use the existing multi_edge path (n_back>=2)
           or sequential per-edge sweeps (n_back<2).
        ---------------------------------------------------------------- */
        int v_idx  = fs - 1;
        size_t states_in = curr->cnt;
        int    n_back    = 0;
        int    widxs[8];

        for (int ai = adj_off[v]; ai < adj_off[v + 1]; ai++) {
            int w = adj_dat[ai];
            if (fidx[w] >= 0 && pos[w] < pos[v] && n_back < 8)
                widxs[n_back++] = fidx[w];
        }

        /* Collect elimination candidates (ascending), then reverse. */
        int elim_idxs_asc[MAX_FS_FAST + 2], n_elim = 0;
        for (int ei = 0; ei < fs; ei++)
            if (last_s[frontier[ei]] <= step)
                elim_idxs_asc[n_elim++] = ei;
        int elim_idxs_desc[MAX_FS_FAST + 2];
        for (int e = 0; e < n_elim; e++)
            elim_idxs_desc[e] = elim_idxs_asc[n_elim - 1 - e];

        /* Choose sweep strategy. */
        int use_fused = (n_elim >= 1) && (curr->cnt >= FUSED_MIN_STATES ||
                                          n_back >= 2);
        /* Also use fused for large tables even with n_elim==0: the sequential
           edge path uses plain ht_add which is not slot_cap-aware.
           When slot_cap > 0 and a large table would be needed, force fused
           so that ht_add_capped is used instead.                             */
        if (!use_fused && slot_cap > 0 && n_back >= 2 && curr->cnt >= FUSED_MIN_STATES)
            use_fused = 1;

        if (use_fused) {
            /* Fused: edges + eliminations in one source-table scan. */
            fused_sweep(curr, nxt, fs, v_idx,
                        widxs, n_back,
                        elim_idxs_desc, n_elim,
                        step, n, &total, slot_cap);
                if (ht_overflow) {
                    fprintf(stderr,
                        "# ERROR: slot cap overflow at step %d "
                        "(table >=90%%%% full).\n"
                        "# Increase --mem-reserve and rerun.\n", step);
                    *res_lo = *res_hi = UINT64_MAX; goto cleanup;
                }
            { HT *tmp = curr; curr = nxt; nxt = tmp; }
        } else {
            /* Separate: sequential edge passes then elimination. */

            /* B: one pass per back-edge. */
            for (int j = 0; j < n_back; j++) {
                int w_idx = widxs[j];
                ht_ensure_clear(nxt, curr->cnt * 2 + 4, slot_cap);
                for (size_t i = 0; i < curr->cap; i++) {
                    if (!curr->keys[i]) continue;
                    u64    key = curr->keys[i];
                    u128   cnt = curr->vals[i];
                    int8_t sv  = slot_get(key, v_idx);
                    int8_t sw  = slot_get(key, w_idx);
                    ht_add(nxt, key, cnt);
                    if (sv == -1 || sw == -1) continue;
                    u64 nk = key;
                    if (sv == 0 && sw == 0) {
                        int8_t L = label_max(key, fs) + 1;
                        nk = slot_set(nk, v_idx, L);
                        nk = slot_set(nk, w_idx, L);
                        ht_add(nxt, canon(nk, fs), cnt);
                    } else if (sv == 0) {
                        nk = slot_set(nk, w_idx, -1);
                        nk = slot_set(nk, v_idx, sw);
                        ht_add(nxt, canon(nk, fs), cnt);
                    } else if (sw == 0) {
                        nk = slot_set(nk, v_idx, -1);
                        nk = slot_set(nk, w_idx, sv);
                        ht_add(nxt, canon(nk, fs), cnt);
                    } else if (sv == sw) {
                        continue;
                    } else {
                        int sv_c = label_count(key, fs, sv);
                        int sw_c = label_count(key, fs, sw);
                        nk = slot_set(nk, v_idx, -1);
                        nk = slot_set(nk, w_idx, -1);
                        nk = label_rename(nk, fs, sw, sv);
                        if (sv_c == 1 && sw_c == 1) {
                            if (nc_get(nk) >= 1) continue;
                            int ok = 1;
                            for (int k2 = 0; k2 < fs; k2++)
                                if (slot_get(nk, k2) != -1) { ok = 0; break; }
                            if (!ok) continue;
                            ht_add(nxt, canon(nc_set_1(nk), fs), cnt);
                        } else {
                            ht_add(nxt, canon(nk, fs), cnt);
                        }
                    }
                }
                { HT *tmp = curr; curr = nxt; nxt = tmp; }
            }

            /* C: eliminate vertices one at a time (ascending order). */
            for (int e = 0; e < n_elim; e++) {
                int u_idx = elim_idxs_asc[e];
                ht_ensure_clear(nxt, curr->cnt + 4, slot_cap);
                for (size_t i = 0; i < curr->cap; i++) {
                    if (!curr->keys[i]) continue;
                    u64    key = curr->keys[i];
                    u128   cnt = curr->vals[i];
                    int8_t su  = slot_get(key, u_idx);
                    int    nc  = nc_get(key);
                    if (su == 0) continue;
                    u64 rk = eliminate_slot(key, u_idx, fs);
                    if (su == -1) {
                        ht_add(nxt, canon(rk, fs - 1), cnt);
                    } else {
                        int partner = 0;
                        for (int k2 = 0; k2 < fs - 1; k2++)
                            if (slot_get(rk, k2) == su) { partner = 1; break; }
                        if (partner) {
                            ht_add(nxt, canon(rk, fs - 1), cnt);
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
                { HT *tmp = curr; curr = nxt; nxt = tmp; }
                fs--;
                /* Shift frontier: remove the eliminated vertex.
                   After removing slot elim_idxs_asc[e], indices for
                   subsequent eliminations shift down by 1, but since
                   elim_idxs_asc is already in ascending order and we
                   process sequentially, u_idx always refers to the
                   current frontier array before this step's shift. */
                int u = frontier[u_idx];
                for (int k = u_idx; k < fs; k++) {
                    frontier[k] = frontier[k + 1];
                    fidx[frontier[k]] = k;
                }
                fidx[u] = -1;
                /* Adjust remaining ascending indices for the shift. */
                for (int e2 = e + 1; e2 < n_elim; e2++)
                    if (elim_idxs_asc[e2] > u_idx)
                        elim_idxs_asc[e2]--;
            }
        }

        /* Update frontier array: remove all eliminated vertices.
           For the fused path: apply the removals now (not done above). */
        if (use_fused) {
            /* Process in descending order: removing high indices first
               doesn't affect the positions of lower-index entries. */
            for (int e = 0; e < n_elim; e++) {
                int u_idx = elim_idxs_desc[e];  /* already adjusted? no - original */
                /* After removing previous (higher) entries, u_idx is still correct
                   because we removed only entries at positions > u_idx. */
                int u = frontier[u_idx];
                for (int k = u_idx; k < fs - 1; k++) {
                    frontier[k] = frontier[k + 1];
                    fidx[frontier[k]] = k;
                }
                fidx[u] = -1;
                fs--;
            }
        }

        /* Collect merge-path completions at the final step. */
        if (step == n - 1) {
            u64 target = KEY_MARKER | KEY_NC_BIT;
            size_t idx = ht_hash(target, curr->cap);
            while (curr->keys[idx] && curr->keys[idx] != target)
                idx = (idx + 1) & (curr->cap - 1);
            if (curr->keys[idx] == target)
                total += curr->vals[idx];
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

        /* Save checkpoint if enough time has elapsed (always, not just verbose). */
        if (checkpoint_path && checkpoint_path[0] && checkpoint_secs > 0) {
            double t_now2 = now_ms();
            if ((t_now2 - t_last_ckpt) / 1000.0 >= checkpoint_secs) {
                if (ckpt_save(checkpoint_path, n, order, step, fs,
                              frontier, total, curr)) {
                    if (verbose)
                        fprintf(stderr, "# Checkpoint saved at step %d "
                                "(%zu states)\n", step, curr->cnt);
                    t_last_ckpt = t_now2;
                }
            }
        }
    }

cleanup:
    ht_free(curr);
    ht_free(nxt);
    free(fidx);

    /* Only write results if we didn't hit the MAX_FS_FAST error path.
       The error path already wrote UINT64_MAX to both outputs; writing
       total (which is 0) on top would silently hide the error. */
    if (*res_lo != UINT64_MAX || *res_hi != UINT64_MAX) {
        *res_lo = (uint64_t) total;
        *res_hi = (uint64_t)(total >> 64);
    }
}
"""

# ---------------------------------------------------------------------------
_LIB_CACHE: dict = {}

def _get_lib():
    src_hash = hashlib.md5(C_SOURCE.encode()).hexdigest()[:12]
    if src_hash in _LIB_CACHE:
        return _LIB_CACHE[src_hash]

    build_dir = os.path.join(tempfile.gettempdir(), f"ham_dp_c_{src_hash}")
    os.makedirs(build_dir, exist_ok=True)
    c_path  = os.path.join(build_dir, "ham_dp.c")
    so_path = os.path.join(build_dir, "ham_dp.so")

    if not os.path.exists(so_path):
        with open(c_path, "w") as f:
            f.write(C_SOURCE)
        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-std=c11",
             "-o", so_path, c_path],
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
            int verbose, size_t max_table_slots,
            const char *checkpoint_path, double checkpoint_secs,
            int load_num, int load_den);
    """)
    lib = ffi.dlopen(so_path)
    _LIB_CACHE[src_hash] = (lib, ffi)
    return lib, ffi


def _available_ram_bytes() -> int:
    """Return an estimate of available (free + reclaimable) RAM in bytes."""
    try:
        with open('/proc/meminfo') as f:
            info = {}
            for line in f:
                k, v = line.split(':', 1)
                info[k.strip()] = int(v.split()[0]) * 1024
        return info.get('MemAvailable', info.get('MemFree', 0))
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    return 0   # 0 = no cap


def _compute_slot_cap(reserve_bytes: int = 2 << 30) -> int:
    """
    Compute a per-table slot cap based on available RAM.

    The DP uses two hash tables (curr and nxt) simultaneously, each at
    24 bytes/slot.  We budget:
        available_ram - reserve_bytes
    split equally between the two tables.

    reserve_bytes : RAM to leave free for the OS, Python interpreter,
                    pathwidth solver residuals, etc.  Default 2 GB.

    Returns max_slots per table (0 = no cap if RAM detection fails).
    """
    avail = _available_ram_bytes()
    if avail <= reserve_bytes:
        return 0   # can't safely cap; let it run and hope for the best
    usable = avail - reserve_bytes
    # Each of the two tables gets half the usable budget
    bytes_per_table = usable // 2
    slots = bytes_per_table // 24   # 24 bytes per slot (u64 key + u128 val)
    # Round DOWN to the largest power of 2 <= slots
    if slots < 16:
        return 0
    p = 1
    while p * 2 <= slots:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# Named load-factor presets: percent → (num, den)
_LOAD_FACTOR_PRESETS = {
    75: (4, 3),
    80: (5, 4),
    85: (6, 5),
    90: (9, 8),
}

def count_hamiltonian_paths_c(n: int, order: list, adj: dict,
                              verbose: bool = False,
                              mem_reserve_gb: float = 2.0,
                              load_factor: int = 75,
                              checkpoint_path: str = "",
                              checkpoint_secs: float = 300.0) -> int:
    """Count undirected Hamiltonian paths in G_n via the optimised C DP.

    Parameters
    ----------
    n                : graph size
    order            : vertex ordering (1-indexed), length n.
    adj              : adjacency dict {v: iterable_of_neighbours} (1-indexed).
    verbose          : if True, print per-step profiling table to stderr.
    mem_reserve_gb   : GB of RAM to reserve for OS/Python/solver overhead.
                       The remaining available RAM is split equally between
                       the two hash tables.  Default 2.0 GB.
    checkpoint_path  : path for checkpoint file ('' = disabled).
                       If a file already exists at this path, the DP resumes
                       from the saved state.  Updated periodically during the
                       run.  Suggested: '/tmp/ham_ckpt_n59.bin'
    checkpoint_secs  : save a checkpoint at most every this many seconds.
                       Default 300 (5 minutes).  Set to 0 to disable.
    """
    lib, ffi = _get_lib()

    # Compute per-table slot cap from available RAM
    max_slots = _compute_slot_cap(reserve_bytes=int(mem_reserve_gb * (1 << 30)))
    lf_num, lf_den = _LOAD_FACTOR_PRESETS.get(load_factor, (4, 3))
    load_num, load_den = lf_num, lf_den
    if verbose and load_factor != 75:
        import sys
        print(f"# Load factor: {load_factor}% ({lf_num}/{lf_den})", file=sys.stderr)
    if verbose and max_slots:
        import sys
        avail_gb = _available_ram_bytes() / (1 << 30)
        cap_gb = max_slots * 24 / (1 << 30)
        print(f"# Memory: {avail_gb:.1f} GB available, "
              f"per-table cap = {max_slots:,} slots = {cap_gb:.2f} GB",
              file=sys.stderr)

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
        c_res_lo, c_res_hi, int(verbose), ffi.cast("size_t", max_slots),
        c_ckpt, ffi.cast("double", checkpoint_secs),
        load_num, load_den,
    )
    lo, hi = int(c_res_lo[0]), int(c_res_hi[0])
    if lo == hi == 0xFFFFFFFFFFFFFFFF:
        raise RuntimeError(
            "Frontier size exceeded MAX_FS_FAST=15 (pathwidth > 15). "
            "The packed uint64 state encoding is limited to 15 frontier slots. "
            "A wider key type would be needed to go further."
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
        t0=time.time(); count=count_hamiltonian_paths_c(n,order,adj)
        print(f"n={n:3d}  max_fw={mx:2d}  ham_paths={count}  ({time.time()-t0:.3f}s)", flush=True)
