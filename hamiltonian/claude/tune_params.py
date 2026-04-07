#!/usr/bin/env python3
"""
tune_params.py — Empirical parameter tuner for ham_dp_c.py

Applies the Aiken et al. PACT '08 pyramid search to the SM_ buffer-size
parameters that control memory-hierarchy utilisation.

Usage:
    python3 tune_params.py [--quick] [--output path/to/ham_dp_c.py]

  --quick   Shorter sweeps (~15 s instead of ~60 s)
  --output  Patch recommended values directly into that file

Key insight from the paper: optimal buffer sizes for software-managed
memory hierarchies lie NEAR THE CAPACITY BOUNDARY of the relevant cache
level, and the search space is SMOOTH (unlike hardware-cache tiling where
conflict misses create high-frequency oscillations).

Parameters tuned, grouped by the memory level they govern:
  A. SM_RBUF_SIZE      L2   write-combining buffers in the radix scatter pass
  B. SM_BB_BUF         L1   block read-ahead in the K-way merge
  C. SM_EXT_STREAM_BUF SLC  per-stream read buffer for ext (SSD-backed) merge
  D. SM_WORKER_CAP     DRAM per-worker flush-buffer size   (analytical)
  E. SM_NTHREADS       CPU  parallel worker count           (measured)
"""

import argparse, cffi, math, os, platform, re
import struct, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument('--quick',  action='store_true')
ap.add_argument('--output', default=None)
ARGS = ap.parse_args()
QUICK = ARGS.quick

ENTRY = 24   # sizeof(SMEntry) = 3 × u64

# ─────────────────────────────────────────────────────────────────────────────
# Compile benchmark library
# ─────────────────────────────────────────────────────────────────────────────
BENCH_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

typedef uint64_t u64;
/* Use three u64 fields instead of __int128 to stay portable */
typedef struct { u64 key, v0, v1; } SME;   /* 24 bytes */

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Sequential read bandwidth of n bytes; best of reps runs.                 */
double bw_read(const char *buf, size_t n, int reps) {
    volatile u64 sink = 0; double best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        const u64 *p = (const u64*)buf; u64 acc = 0;
        for (size_t i = 0; i < n/8; i++) acc ^= p[i];
        sink ^= acc;
        double bw = (double)n / (now_s()-t0);
        if (bw > best) best = bw;
    }
    (void)sink; return best;
}

/* RadixBuf scatter: 256-bucket scatter with rbuf_size-entry write buffers.
   Measures the flush phase of sm_radix_sort.  Returns entries/second.     */
double bench_rbuf(const char *src_raw, size_t n, char *dst_raw,
                  int rbuf_size, int n_buckets) {
    const SME *src = (const SME*)src_raw;
    SME       *dst = (SME*)dst_raw;
    SME    *slots  = (SME*)   malloc((size_t)n_buckets*rbuf_size*sizeof(SME));
    size_t *cnt    = (size_t*)calloc(n_buckets, sizeof(size_t));
    size_t *prefix = (size_t*)calloc(n_buckets, sizeof(size_t));
    if (!slots||!cnt||!prefix){free(slots);free(cnt);free(prefix);return 0;}
    for (size_t i=0; i<n; i++) cnt[(int)(src[i].key&(n_buckets-1))]++;
    size_t p=0;
    for (int b=0; b<n_buckets; b++){prefix[b]=p;p+=cnt[b];cnt[b]=0;}
    double t0 = now_s();
    for (size_t i=0; i<n; i++) {
        int b=(int)(src[i].key&(n_buckets-1)), ci=(int)cnt[b];
        slots[(size_t)b*rbuf_size+ci]=src[i]; ci++;
        if (ci==rbuf_size) {
            memcpy(dst+prefix[b], slots+(size_t)b*rbuf_size,
                   (size_t)rbuf_size*sizeof(SME));
            prefix[b]+=rbuf_size; ci=0;
        }
        cnt[b]=(size_t)ci;
    }
    for (int b=0;b<n_buckets;b++)
        if (cnt[b]) memcpy(dst+prefix[b],slots+(size_t)b*rbuf_size,
                           cnt[b]*sizeof(SME));
    double elapsed=now_s()-t0;
    free(slots);free(cnt);free(prefix);
    return (double)n/elapsed;
}

/* Simple heap */
typedef struct {u64 key; int s;} HE;
static void hsift(HE*h,int n,int i){
    for(;;){int m=i,l=2*i+1,r=2*i+2;
        if(l<n&&h[l].key<h[m].key)m=l;
        if(r<n&&h[r].key<h[m].key)m=r;
        if(m==i)break; HE t=h[i];h[i]=h[m];h[m]=t;i=m;}
}

/* Block-buffered K-way merge: K DRAM-resident sorted streams.
   Returns entries/second.                                                   */
typedef struct {
    const SME *base; size_t pos,len;
    SME *blk; int bpos,blen,bb;
} BS;

static void bfill(BS*s){
    size_t rem=s->len-s->pos;
    int n=(int)(rem<(size_t)s->bb?rem:(size_t)s->bb);
    memcpy(s->blk,s->base+s->pos,(size_t)n*sizeof(SME));
    s->pos+=n;s->bpos=0;s->blen=n;
}
static int bempty(BS*s){
    if(s->bpos<s->blen)return 0;
    if(s->pos>=s->len)return 1;
    bfill(s);return s->blen==0;
}

double bench_merge_bb(const char *data_raw, size_t stream_len, int K,
                      char *out_raw, int bb_buf) {
    const SME *data=(const SME*)data_raw;
    SME       *out =(SME*)out_raw;
    BS  *S=(BS*)  malloc((size_t)K*sizeof(BS));
    HE  *h=(HE*)  malloc((size_t)K*sizeof(HE));
    SME**bufs=(SME**)malloc((size_t)K*sizeof(SME*));
    for (int i=0;i<K;i++){
        bufs[i]=(SME*)malloc((size_t)bb_buf*sizeof(SME));
        S[i]=(BS){data+(size_t)i*stream_len,0,stream_len,bufs[i],0,0,bb_buf};
        bfill(&S[i]);
    }
    int hs=0;
    for(int i=0;i<K;i++) if(!bempty(&S[i])){h[hs].key=S[i].blk[S[i].bpos].key;h[hs].s=i;hs++;}
    for(int i=hs/2-1;i>=0;i--) hsift(h,hs,i);
    double t0=now_s();
    size_t op=0;
    while(hs>0){
        int s=h[0].s; out[op++]=S[s].blk[S[s].bpos++];
        if(!bempty(&S[s])){h[0].key=S[s].blk[S[s].bpos].key;hsift(h,hs,0);}
        else{h[0]=h[--hs];if(hs>0)hsift(h,hs,0);}
    }
    double elapsed=now_s()-t0;
    for(int i=0;i<K;i++) free(bufs[i]);
    free(S);free(h);free(bufs);
    return (double)(K*stream_len)/elapsed;
}

/* Parallel compute throughput: nt threads, each doing work_pp operations.  */
typedef struct {size_t n;int id;volatile u64*sink;double res;} TA;
static void*tw(void*arg){
    TA*a=(TA*)arg; u64 acc=(u64)a->id;
    for(size_t i=0;i<a->n;i++){acc^=acc>>17;acc*=0xbf58476d1ce4e5b9ULL;acc^=acc>>31;}
    a->sink[a->id]=acc; a->res=(double)a->n; return NULL;
}
void bench_threads(int max_threads, size_t work_pp, double *out) {
    volatile u64 *sink=(volatile u64*)calloc((size_t)max_threads,sizeof(u64));
    TA  *args=(TA*) malloc((size_t)max_threads*sizeof(TA));
    pthread_t*thr=(pthread_t*)malloc((size_t)max_threads*sizeof(pthread_t));
    for(int nt=1;nt<=max_threads;nt++){
        for(int i=0;i<nt;i++) args[i]=(TA){work_pp,i,sink,0};
        double t0=now_s();
        for(int i=0;i<nt;i++) pthread_create(&thr[i],NULL,tw,&args[i]);
        for(int i=0;i<nt;i++) pthread_join(thr[i],NULL);
        out[nt-1]=(double)(nt*work_pp)/(now_s()-t0);
    }
    free((void*)sink);free(args);free(thr);
}
"""

print("Compiling benchmarks...", end=' ', flush=True)
ffi = cffi.FFI()
ffi.cdef("""
double bw_read    (const char*,size_t,int);
double bench_rbuf (const char*,size_t,char*,int,int);
double bench_merge_bb(const char*,size_t,int,char*,int);
void   bench_threads (int,size_t,double*);
""")
_td = tempfile.mkdtemp()
_cf = _td+'/bench.c'; _lf = _td+'/bench.so'
with open(_cf,'w') as f: f.write(BENCH_C)
_r = subprocess.run(['gcc','-O3','-march=native','-shared','-fPIC',
                     '-pthread','-o',_lf,_cf],capture_output=True,text=True)
if _r.returncode:
    print("FAIL\n",_r.stderr); sys.exit(1)
lib = ffi.dlopen(_lf)
print("OK")

# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────
def _sysctl(key):
    try:
        r=subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
        return int(r.stdout.strip())
    except: return None

def _linux_cache(level, typ='Data'):
    base='/sys/devices/system/cpu/cpu0/cache'
    try:
        for idx in range(10):
            d=f'{base}/index{idx}'
            if not os.path.isdir(d): break
            t=open(f'{d}/type').read().strip()
            lv=int(open(f'{d}/level').read())
            if lv==level and(typ in t or t=='Unified'):
                s=open(f'{d}/size').read().strip()
                m={'K':1024,'M':1<<20,'G':1<<30}.get(s[-1],1)
                return int(s[:-1])*m if s[-1].isalpha() else int(s)
    except: pass
    return None

SYS = platform.system()
if SYS=='Darwin':
    P_CORES = _sysctl('hw.perflevel0.physicalcpu') or os.cpu_count()//2
    E_CORES = _sysctl('hw.perflevel1.physicalcpu') or 0
    L1      = _sysctl('hw.perflevel0.l1dcachesize') or (192<<10)
    L2      = _sysctl('hw.perflevel0.l2cachesize')  or (4<<20)
    SLC     = _sysctl('hw.l3cachesize')              or (12<<20)
    RAM     = _sysctl('hw.memsize')                  or (16<<30)
    CPU     = subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],
                             capture_output=True,text=True).stdout.strip()
else:
    P_CORES = os.cpu_count() or 4; E_CORES = 0
    L1  = _linux_cache(1) or (32<<10)
    L2  = _linux_cache(2) or (256<<10)
    SLC = _linux_cache(3) or _linux_cache(2) or (8<<20)
    try:
        with open('/proc/meminfo') as f:
            RAM=next(int(l.split()[1])*1024 for l in f if l.startswith('MemTotal'))
    except: RAM=8<<30
    try:
        with open('/proc/cpuinfo') as f:
            CPU=next(l.split(':')[1].strip() for l in f if 'model name' in l)
    except: CPU='unknown'

print(f"\n{'='*62}")
print("Stage 1: Hardware")
print(f"{'='*62}")
print(f"  CPU:    {CPU[:55]}")
print(f"  Cores:  {P_CORES} P-cores  {E_CORES} E-cores")
print(f"  Cache:  L1={L1>>10}KB  L2={L2>>10}KB  SLC={SLC>>20}MB")
print(f"  RAM:    {RAM>>30}GB")

# Empirical bandwidth sweep to locate cache boundaries
def bw_sweep():
    sizes, sz = [], 4096
    cap = min(64<<20, RAM//8)
    while sz <= cap: sizes.append(sz); sz=int(sz*1.5)
    if QUICK: sizes=[s for s in sizes if s<=(32<<20)]
    reps = 2 if QUICK else 4
    ba   = bytearray(max(sizes))
    buf  = ffi.from_buffer(ba)
    return [(sz, lib.bw_read(buf, sz, reps)/1e9) for sz in sizes]

print("\n  Bandwidth sweep...", end=' ', flush=True)
bw = bw_sweep(); print("done")

def _bw_at(lo, hi):
    v=[b for s,b in bw if lo<=s<=hi]; return max(v) if v else 0

BW_L1   = _bw_at(0,       L1*2)
BW_L2   = _bw_at(L1*2,    L2*2)
BW_SLC  = _bw_at(L2*2,    SLC*2)
BW_DRAM = _bw_at(SLC*2,   RAM//2) or bw[-1][1]
print(f"  Bandwidth: L1={BW_L1:.0f}  L2={BW_L2:.0f}  "
      f"SLC={BW_SLC:.0f}  DRAM={BW_DRAM:.0f}  GB/s")

# Helper: allocate aligned benchmark buffers
def _buf(n_entries):
    return ffi.new(f'char[{n_entries*ENTRY}]')

def _fill_keys(buf, n):
    """Fill buf with SME entries with pseudo-random keys."""
    view = ffi.buffer(buf, n*ENTRY)
    ba = memoryview(view).cast('B')
    for i in range(n):
        k = ((i*1234567891011)^(i>>7)) & 0xFFFFFFFFFFFFFFFF
        struct.pack_into('<QQQ', ba, i*ENTRY, k, i, 0)

def _fill_sorted_streams(buf, stream_len, K):
    """Fill K sorted streams with interleaved keys."""
    view = ffi.buffer(buf, K*stream_len*ENTRY)
    ba = memoryview(view).cast('B')
    for k in range(K):
        for i in range(stream_len):
            struct.pack_into('<QQQ', ba, (k*stream_len+i)*ENTRY, K*i+k, 0, 0)

def _measure(fn, reps):
    """Return best throughput over reps runs."""
    best = 0
    for _ in range(reps):
        v = fn()
        if v > best: best = v
    return best

REPS = 2 if QUICK else 3

# ─────────────────────────────────────────────────────────────────────────────
# Group A: SM_RBUF_SIZE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("Stage 2a: SM_RBUF_SIZE  (radix scatter write-combining)")
print(f"{'='*62}")
print("  Constraint: 256 × SM_RBUF_SIZE × 24B ≤ L2")
print("  Paper: best value near capacity boundary → fill L2")

N_A = max(2_000_000, min(10_000_000, L2*8//ENTRY))
if QUICK: N_A = min(N_A, 2_000_000)
src_A = _buf(N_A); dst_A = _buf(N_A)
_fill_keys(src_A, N_A)
# warm up
lib.bench_rbuf(src_A, N_A, dst_A, 32, 256)

cands_A = []
v=8
while v*256*ENTRY <= L2*2: cands_A.append(v); v*=2

print(f"\n  N={N_A//1_000_000}M entries, 256 buckets, L2={L2>>10}KB")
print(f"  {'RBUF':>6}  {'pool KB':>8}  {'in L2':>6}  {'Mentry/s':>10}  {'GB/s':>6}")

res_A = []
for val in cands_A:
    pool_kb = val*256*ENTRY/1024
    fits    = pool_kb <= L2/1024
    t = _measure(lambda v=val: lib.bench_rbuf(src_A,N_A,dst_A,v,256), REPS)
    res_A.append((val, t))
    mk = ' ←' if len(res_A)>1 and t>max(r[1] for r in res_A[:-1]) else ''
    print(f"  {val:>6}  {pool_kb:>8.0f}  {'yes' if fits else 'no':>6}  "
          f"{t/1e6:>10.1f}  {t*ENTRY/1e9:>6.2f}{mk}")

in_l2_A = [(v,t) for v,t in res_A if v*256*ENTRY<=L2]
rec_A = max(in_l2_A, key=lambda r:r[1])[0] if in_l2_A else res_A[0][0]
print(f"\n  → Recommended SM_RBUF_SIZE = {rec_A}")

# ─────────────────────────────────────────────────────────────────────────────
# Group B: SM_BB_BUF
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("Stage 2b: SM_BB_BUF  (block read-ahead in K-way merge)")
print(f"{'='*62}")
print("  Constraint: P × SM_BB_BUF × 24B ≤ L1")
print("  Optimal: fill L1 (Aiken capacity-boundary principle).")
print()
print("  SM_BB_BUF is computed analytically, not measured.")
print("  The benchmark for this parameter is unreliable: it runs after the")
print("  radix benchmark which warms the cache, making small buffers appear")
print("  equivalent to large ones when the streams are still SLC-resident.")
print("  Empirically, 40% performance differences have been observed between")
print("  values that the benchmark rated identically.")
print()

# Analytical formula: largest power-of-2 such that K × buf × ENTRY ≤ L1
# K = rec_D (P-cores, already determined)
_bb_raw = L1 // (P_CORES * ENTRY)         # max entries fitting in L1; K = P_CORES
rec_B = 1
while rec_B * 2 <= _bb_raw: rec_B *= 2   # round down to power-of-2
rec_B = max(16, min(rec_B, 4096))         # clamp: floor 16, ceil 4096

print(f"  L1={L1>>10}KB  K={P_CORES} P-cores  ENTRY=24B")
print(f"  Max pool: L1 / (K × 24) = {L1>>10}KB / ({P_CORES}×24) "
      f"= {_bb_raw} entries → rounded to {rec_B}")
print(f"  Pool size: {P_CORES} × {rec_B} × 24 = {P_CORES*rec_B*24//1024}KB "
      f"({'< L1 ✓' if P_CORES*rec_B*ENTRY <= L1 else '> L1 ✗'})")

# Verification benchmark (informational only — does not change rec_B)
print()
print("  Verification benchmark (informational; does not override formula):")
K_B = P_CORES
stream_B = max(200_000, min(1_000_000, SLC*2//(K_B*ENTRY)))
if QUICK: stream_B = min(stream_B, 200_000)
src_B = _buf(K_B*stream_B); out_B = _buf(K_B*stream_B)
_fill_sorted_streams(src_B, stream_B, K_B)
# Flush caches by touching a large unrelated array before measuring
_flush_arr = _buf(min(SLC*4, 64<<20))
_flush_view = ffi.buffer(_flush_arr, min(SLC*4, 64<<20))
_dummy = bytes(_flush_view[:64])  # force allocation
del _flush_arr

base_B = _measure(lambda: lib.bench_merge_bb(src_B,stream_B,K_B,out_B,1), 2)
print(f"  {'BB_BUF':>6}  {'pool KB':>7}  {'in L1':>6}  {'Mentry/s':>10}  {'speedup':>8}")
for val in [16, 32, 64, 128, 256, rec_B]:
    val = min(val, 4096)
    pool_kb = val*K_B*ENTRY/1024
    fits    = val*K_B*ENTRY <= L1
    t = _measure(lambda v=val: lib.bench_merge_bb(src_B,stream_B,K_B,out_B,v), 2)
    mk = ' ← formula' if val == rec_B else ''
    print(f"  {val:>6}  {pool_kb:>7.1f}  {'yes' if fits else 'no':>6}  "
          f"{t/1e6:>10.1f}  {t/base_B:>7.1f}x{mk}")

print(f"\n  → Recommended SM_BB_BUF = {rec_B}  (analytical: L1 / (P × 24), pow2)"      f"  SM_BB_KSTACK = {max(rec_B, 32)}")

# ─────────────────────────────────────────────────────────────────────────────
# Group C: SM_EXT_STREAM_BUF
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("Stage 2c: SM_EXT_STREAM_BUF  (ext K-way merge stream buffer)")
print(f"{'='*62}")
print("  Constraint: K_total × buf_bytes ≤ SLC")
print("  Paper: best value near SLC capacity boundary.")
print("  NOTE: the optimal value depends on K_total (= P × n_runs_per_worker),")
print("  which varies per step.  We derive a formula and a fixed fallback.")

K_cases  = [6,12,24,48] if not QUICK else [6,24]
buf_vals = []
v=256
while v*max(K_cases)*ENTRY <= SLC*3: buf_vals.append(v); v*=2

print(f"\n  SLC={SLC>>20}MB. Testing K={K_cases} × buf_size:")
print(f"  {'K':>4}  {'buf':>6}  {'total MB':>9}  {'SLC%':>6}  {'Mentry/s':>10}")

ratios = []
for K in K_cases:
    stream_C = max(100_000, SLC//(K*ENTRY)*3)
    if QUICK: stream_C=max(100_000, stream_C//2)
    src_C = _buf(K*stream_C); out_C = _buf(K*stream_C)
    _fill_sorted_streams(src_C, stream_C, K)
    best_buf, best_t = 1, 0
    for buf in buf_vals:
        total_mb = K*buf*ENTRY/(1<<20)
        slc_pct  = K*buf*ENTRY/SLC*100
        t = _measure(lambda b=buf: lib.bench_merge_bb(src_C,stream_C,K,out_C,b), REPS)
        mk=''
        if t > best_t: best_t=t; best_buf=buf; mk=' ←'
        print(f"  {K:>4}  {buf:>6}  {total_mb:>9.1f}  "
              f"{slc_pct:>5.0f}%  {t/1e6:>10.1f}{mk}")
    formula = SLC//(K*ENTRY)
    ratio   = best_buf/formula
    ratios.append(ratio)
    print(f"       best={best_buf}, formula(SLC/K/24)={formula}, ratio={ratio:.2f}")

mean_ratio = sum(ratios)/len(ratios)
print(f"\n  Mean ratio empirical/formula = {mean_ratio:.2f}")
adj = f" × {mean_ratio:.1f}" if abs(mean_ratio-1)>0.2 else ""
print(f"  Dynamic formula: SM_EXT_STREAM_BUF = SLC_bytes{adj} / (K_total × 24)")

# Fixed fallback for small K (early steps with few runs)
rec_C = max(256, int(SLC * mean_ratio / (6 * ENTRY)))
rec_C = 2**round(math.log2(rec_C))
print(f"  Fixed fallback (K≤6): {rec_C} entries  "
      f"= {rec_C*ENTRY//1024}KB each, {6*rec_C*ENTRY//1024}KB total")
print(f"\n  → Recommended SM_EXT_STREAM_BUF (fixed) = {rec_C}")
print(f"  → Dynamic formula: SLC_bytes / (P × n_runs × 24)")

# ─────────────────────────────────────────────────────────────────────────────
# Group D: SM_NTHREADS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("Stage 2d: SM_NTHREADS  (parallel worker count)")
print(f"{'='*62}")
print("  NTHREADS is not a buffer-size tunable — it has a hardware-defined")
print("  right answer: the P-core count.  We use OS-reported P_CORES as the")
print("  primary value and run the benchmark only to verify and report scaling.")
print("  The efficiency threshold is intentionally NOT used to override P_CORES")
print("  because short benchmark runs are too noisy (scheduling jitter, thermal")
print("  state, E-core interference) to reliably distinguish 2 from 4 threads.")

max_t = min(P_CORES+E_CORES, 16)
wpp   = 5_000_000 if not QUICK else 1_000_000
# Run benchmark 3 times and take the best to reduce jitter
tputs = ffi.new(f'double[{max_t}]')
best_tputs = [0.0] * max_t
for _run in range(3 if not QUICK else 2):
    lib.bench_threads(max_t, wpp, tputs)
    for i in range(max_t):
        if tputs[i] > best_tputs[i]: best_tputs[i] = tputs[i]

print(f"\n  {'threads':>8}  {'Mops/s':>8}  {'speedup':>8}  {'efficiency':>10}")
base_t = best_tputs[0]
for nt in range(1, max_t+1):
    t   = best_tputs[nt-1]
    sp  = t/base_t
    eff = sp/nt*100
    is_p = (nt <= P_CORES)
    mk  = ' (P-core)' if is_p else ' (E-core)'
    print(f"  {nt:>8}  {t/1e6:>8.1f}  {sp:>8.2f}x  {eff:>9.0f}%{mk}")

# Primary recommendation: all P-cores.
# Only reduce if the benchmark shows negative scaling (pathological case,
# e.g. NUMA or heavily hyperthreaded machine where P_CORES is overcounted).
rec_D = P_CORES
best_P = max(range(1, P_CORES+1), key=lambda n: best_tputs[n-1])
if best_P < P_CORES:
    print(f"\n  ⚠  Benchmark peak at {best_P} threads < P_CORES ({P_CORES}).")
    print(f"     This may indicate thermal throttling or OS detection error.")
    print(f"     Using P_CORES={P_CORES}; override manually in machine.yaml if needed.")
print(f"\n  OS-reported P-cores: {P_CORES}  E-cores: {E_CORES}")
print(f"  Benchmark peak:      {best_P} threads")
print(f"  → Recommended SM_NTHREADS = {rec_D}  (= P_CORES, hardware-defined)")

# ─────────────────────────────────────────────────────────────────────────────
# Group E: SM_WORKER_CAP  (analytical)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("Stage 2e: SM_WORKER_CAP  (per-worker flush buffer, analytical)")
print(f"{'='*62}")

OS_MARGIN = 4<<30
# Leave RAM for curr + nxt (≈ 40% of RAM for typical large step)
# Worker bufs = P × 2 × CAP × 24B
avail = RAM - OS_MARGIN - int(0.4*RAM)
cap_max = avail // (rec_D * 2 * ENTRY)
cap_p2  = 1
while cap_p2*2 <= cap_max: cap_p2 *= 2
cap_p2  = max(4<<20, cap_p2)

print(f"\n  RAM={RAM>>30}GB, P={rec_D}, OS+margin={OS_MARGIN>>30}GB")
print(f"  Budget for worker bufs ≈ {avail>>30}GB")
print(f"\n  {'CAP (M)':>8}  {'bufs GB':>8}  {'runs/w@1B':>10}  {'fits RAM':>9}")
for cm in [4,8,16,32,64]:
    cap  = cm<<20
    bg   = rec_D*2*cap*ENTRY/(1<<30)
    runs = math.ceil(1_000_000_000/(rec_D*cap))
    fits = bg + OS_MARGIN/(1<<30) + RAM*0.4/(1<<30) <= RAM/(1<<30)
    mk   = ' ←' if cap==cap_p2 else ''
    print(f"  {cm:>8}  {bg:>8.1f}  {runs:>10}  {'yes' if fits else 'no':>9}{mk}")

print(f"\n  → Recommended SM_WORKER_CAP = {cap_p2>>20}M entries "
      f"({cap_p2*ENTRY>>20}MB per buffer)")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
rec = {
    'SM_NTHREADS':       rec_D,
    'SM_WORKER_CAP_M':   cap_p2>>20,
    'SM_RBUF_SIZE':      rec_A,
    'SM_BB_BUF':         rec_B,
    'SM_BB_KSTACK':      max(rec_B, 32),
    'SM_EXT_STREAM_BUF': rec_C,
    'SM_SLC_BYTES_MB':   SLC>>20,
}

print(f"\n{'='*62}")
print("Stage 3: Recommendations")
print(f"{'='*62}")
print(f"""
  /* Tuned for: {CPU[:50]} */
  /* RAM: {RAM>>30}GB  L1:{L1>>10}KB  L2:{L2>>10}KB  SLC:{SLC>>20}MB */

  #define SM_NTHREADS        {rec['SM_NTHREADS']}
  #define SM_WORKER_CAP      ({rec['SM_WORKER_CAP_M']} * 1024 * 1024)
  #define SM_RBUF_SIZE       {rec['SM_RBUF_SIZE']}
  #define SM_BB_BUF          {rec['SM_BB_BUF']}
  #define SM_BB_KSTACK       {rec['SM_BB_KSTACK']}
  /* Fixed fallback; dynamic formula: SLC_bytes / (P*n_runs*24) */
  #define SM_EXT_STREAM_BUF  {rec['SM_EXT_STREAM_BUF']}
  #define SM_SLC_BYTES       ({SLC})   /* {SLC>>20}MB */
""")

DEFS = [
    ('SM_NTHREADS',     6),
    ('SM_WORKER_CAP_M', 32),
    ('SM_RBUF_SIZE',    32),
    ('SM_BB_BUF',       32),
    ('SM_EXT_STREAM_BUF', 16384),
    ('SM_SLC_BYTES_MB', 12),
]
print("  Changes from ham_dp_c.py defaults:")
changed = False
for name, default in DEFS:
    key   = name.replace('_CAP_M','_CAP').replace('_BYTES_MB','_BYTES')
    new   = rec.get(name) or rec.get(name+'_M')
    if new != default:
        changed = True
        print(f"    {name:<22}: {default} → {new}")
if not changed:
    print("    (defaults are already optimal for this machine)")

# ─────────────────────────────────────────────────────────────────────────────
# Patch file
# ─────────────────────────────────────────────────────────────────────────────
if ARGS.output:
    # Write machine.yaml next to the specified file (or current dir).
    import pathlib, yaml as _yaml
    out_dir  = pathlib.Path(ARGS.output).parent
    yaml_path = out_dir / "machine.yaml"
    yaml_data = {
        # Tuning constants: written by tune_params.py, read by _derive_build_constants.
        # All SM_ values are passed to gcc via -D flags; none are hardcoded in C.
        "SM_NTHREADS":         rec_D,
        "SM_WORKER_CAP":       cap_p2,            # entries (not MB)
        "SM_RBUF_SIZE":        rec_A,
        "SM_BB_BUF":           rec_B,
        "SM_BB_KSTACK":        max(rec_B, 32),
        "SM_EXT_STREAM_BUF":   rec_C,
        "SM_SLC_BYTES":        SLC,               # bytes
        "SM_PAR_MERGE_THRESH": 50_000_000,
        "SM_STEAL_FACTOR":     32,
        # Hardware info (informational; not used by build)
        "_cpu":    CPU[:60],
        "_ram_gb": RAM >> 30,
        "_l1_kb":  L1  >> 10,
        "_l2_kb":  L2  >> 10,
        "_slc_mb": SLC >> 20,
    }
    with open(yaml_path, "w") as _f:
        _yaml.dump(yaml_data, _f, default_flow_style=False, sort_keys=True)
    print(f"\n" + "="*62)
    print(f"Wrote {yaml_path}")
    print("="*62)
    for k, v in yaml_data.items():
        if not k.startswith("_"):
            print(f"  {k:<24} = {v}")
    print()
    print("  ham_dp_c.py will pick up these values automatically at")
    print("  next build (no patching of C source needed).")

print("\nDone.")
