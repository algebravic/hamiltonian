/* cacheprobe.c — report and measure cache geometry for ham_dp_c tuning.
 *
 * Prints YAML on stdout.  Two parts:
 *   os:        what the OS reports (Linux sysfs, macOS sysctl)
 *   measured:  what timing experiments show
 *
 * Measurements (all user-space, no root):
 *   coherence  Two threads on different cores increment counters d bytes
 *              apart.  Throughput collapses while both counters share a
 *              coherence unit; the granularity is the smallest d at which
 *              throughput recovers.  Catches 128 B adjacent-line pairing.
 *   l1         Pointer chase over k lines spaced S = 2^j bytes apart, in
 *              random cyclic order.  If S is a multiple of the L1 way size
 *              W all k lines fall in one set, and latency jumps when
 *              k > ways.  L1 is virtually indexed within the page, so
 *              virtual strides are exact.  Gives ways, W, sets = W / line.
 *   l2         Minimal "thrashing set" by group testing (after Vila, Koepf,
 *              Morales, IEEE S&P 2019).  Candidates are lines with equal
 *              page offset; a set thrashes if chasing it repeatedly runs
 *              at below-L2 speed.  For LRU-like replacement the minimal
 *              thrashing set has ways + 1 lines.  Repeated for several
 *              targets; reports the mode and the fraction agreeing.  With
 *              non-LRU replacement the answer can be fuzzy: low confidence
 *              is reported as such, never rounded to a guess.
 *   latency    Random chase over working sets 16 KiB .. 1 GiB (bounded by
 *              RAM/8); knees where latency rises by > 1.5x.
 *   dtlb       Chase one line per page over k pages; knees give TLB reach.
 *   dczva      AArch64 only: DC ZVA block size from DCZID_EL0.
 *
 * Usage: cacheprobe [--quick] [--skip-l2]
 * Build: cc -O2 -pthread -o cacheprobe cacheprobe.c
 * Caveat: in a VM the coherence and L2 experiments can fail; they then
 *         print null rather than a guess.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#ifdef __APPLE__
#  include <sys/sysctl.h>
#  include <pthread/qos.h>
#endif
#ifdef __linux__
#  include <sched.h>
#endif

#ifndef MAP_ANONYMOUS
#  define MAP_ANONYMOUS MAP_ANON
#endif

static int QUICK = 0;

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) {
    uint64_t x = rng_state; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}

static void *big_alloc(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(p, 1, bytes);                       /* fault in */
    return p;
}

static void pin_to(int cpu) {
#ifdef __linux__
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
#elif defined(__APPLE__)
    (void)cpu;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  /* prefer P-cores */
#else
    (void)cpu;
#endif
}

/* ── pointer chase: link addrs[0..k) into one random cycle, return ns/load ─ */
static void link_cycle(char **addrs, int k) {
    for (int i = k - 1; i > 0; i--) {          /* Fisher–Yates */
        int j = (int)(rnd() % (uint64_t)(i + 1));
        char *t = addrs[i]; addrs[i] = addrs[j]; addrs[j] = t;
    }
    for (int i = 0; i < k; i++) *(char **)addrs[i] = addrs[(i + 1) % k];
}

static volatile uintptr_t sink;
static double chase(char *start, long loads) {
    char *p = start;
    for (long i = 0; i < 1000; i++) p = *(char **)p;   /* warm */
    double t0 = now_ns();
    for (long i = 0; i < loads; i++) p = *(char **)p;
    double t1 = now_ns();
    sink = (uintptr_t)p;
    return (t1 - t0) / loads;
}

static double chase_best(char *start, long loads, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) { double v = chase(start, loads); if (v < best) best = v; }
    return best;
}

/* ── OS-reported geometry ─────────────────────────────────────────────── */
typedef struct { long size, line, ways, sets, shared_by; } CacheLvl;
static CacheLvl os_l[4];                       /* 1 = L1d, 2 = L2, 3 = L3 */
static long page_bytes, line_os = 64, ncpu_logical = 1;

#ifdef __linux__
static long read_long(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char buf[128]; long v = -1;
    if (fgets(buf, sizeof buf, f)) {
        char *e; v = strtol(buf, &e, 10);
        if (*e == 'K') v <<= 10; else if (*e == 'M') v <<= 20; else if (*e == 'G') v <<= 30;
        if (e == buf) v = -1;
    }
    fclose(f); return v;
}
static long count_cpu_list(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char buf[4096]; long n = 0;
    if (fgets(buf, sizeof buf, f)) {
        char *s = buf;
        while (*s && *s != '\n') {
            long a = strtol(s, &s, 10), b = a;
            if (*s == '-') b = strtol(s + 1, &s, 10);
            n += b - a + 1;
            if (*s == ',') s++;
        }
    }
    fclose(f); return n;
}
static int first_non_sibling_cpu(void) {
    /* first CPU not sharing cpu0's L2 (i.e. a different physical core) */
    char path[256];
    for (int idx = 0; idx < 10; idx++) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        if (read_long(path) == 2) {
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/shared_cpu_list", idx);
            FILE *f = fopen(path, "r"); if (!f) break;
            char buf[4096]; int used[4096] = {0};
            if (fgets(buf, sizeof buf, f)) {
                char *s = buf;
                while (*s && *s != '\n') {
                    long a = strtol(s, &s, 10), b = a;
                    if (*s == '-') b = strtol(s + 1, &s, 10);
                    for (long c = a; c <= b && c < 4096; c++) used[c] = 1;
                    if (*s == ',') s++;
                }
            }
            fclose(f);
            for (int c = 1; c < ncpu_logical && c < 4096; c++) if (!used[c]) return c;
            break;
        }
    }
    return ncpu_logical > 1 ? 1 : 0;
}
static void os_geometry(void) {
    char path[256];
    for (int idx = 0; idx < 10; idx++) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        long lv = read_long(path); if (lv < 0) break;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        FILE *f = fopen(path, "r"); char t[32] = "";
        if (f) { if (!fgets(t, sizeof t, f)) t[0] = 0; fclose(f); }
        if (lv == 1 && strncmp(t, "Data", 4)) continue;
        if (lv < 1 || lv > 3) continue;
        CacheLvl *c = &os_l[lv];
#define RD(field, name) snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/" name, idx); c->field = read_long(path)
        RD(size, "size"); RD(line, "coherency_line_size");
        RD(ways, "ways_of_associativity"); RD(sets, "number_of_sets");
#undef RD
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/shared_cpu_list", idx);
        c->shared_by = count_cpu_list(path);
        if (c->line > 0) line_os = c->line;
    }
}
#elif defined(__APPLE__)
static long sysctl_long(const char *name) {
    int64_t v = 0; size_t len = sizeof v;
    if (sysctlbyname(name, &v, &len, NULL, 0) != 0) return -1;
    if (len == 4) return (long)*(int32_t *)&v;
    return (long)v;
}
static int first_non_sibling_cpu(void) { return 1; }
static void os_geometry(void) {
    long l1 = sysctl_long("hw.perflevel0.l1dcachesize");
    if (l1 < 0) l1 = sysctl_long("hw.l1dcachesize");
    long l2 = sysctl_long("hw.perflevel0.l2cachesize");
    if (l2 < 0) l2 = sysctl_long("hw.l2cachesize");
    long cl = sysctl_long("hw.cachelinesize");
    long per = sysctl_long("hw.perflevel0.cpusperl2");
    if (cl > 0) line_os = cl;
    os_l[1] = (CacheLvl){ l1, cl, -1, -1, 1 };
    os_l[2] = (CacheLvl){ l2, cl, -1, -1, per };
    os_l[3] = (CacheLvl){ sysctl_long("hw.l3cachesize"), cl, -1, -1, sysctl_long("hw.logicalcpu") };
}
#else
static int first_non_sibling_cpu(void) { return 1; }
static void os_geometry(void) {}
#endif

/* ── 1. coherence granularity ─────────────────────────────────────────── */
typedef struct { volatile uint64_t *ctr; long iters; int cpu; volatile int *go; double ns; } CohArg;
static void *coh_thread(void *a_) {
    CohArg *a = (CohArg *)a_;
    pin_to(a->cpu);
    while (!*a->go) ;
    double t0 = now_ns();
    for (long i = 0; i < a->iters; i++) (*a->ctr)++;
    a->ns = now_ns() - t0;
    return NULL;
}
static void measure_coherence(int other_cpu, int *gran, double curve[10], int dvals[10], int *nd) {
    char *buf = big_alloc(1 << 16);
    char *base = (char *)(((uintptr_t)buf + 4095) & ~(uintptr_t)4095);
    long iters = QUICK ? 5000000 : 20000000;
    int ds[] = { 8, 16, 32, 64, 128, 256, 512, 1024 };
    int n = 8;
    for (int i = 0; i < n; i++) {
        double best = 0;
        for (int r = 0; r < 3; r++) {
            volatile int go = 0;
            CohArg A = { (volatile uint64_t *)base, iters, 0, &go, 0 };
            CohArg B = { (volatile uint64_t *)(base + ds[i]), iters, other_cpu, &go, 0 };
            *A.ctr = 0; *B.ctr = 0;
            pthread_t ta, tb;
            pthread_create(&ta, NULL, coh_thread, &A);
            pthread_create(&tb, NULL, coh_thread, &B);
            go = 1;
            pthread_join(ta, NULL); pthread_join(tb, NULL);
            double ns = A.ns > B.ns ? A.ns : B.ns;
            double mops = 2.0 * iters / ns * 1e3;
            if (mops > best) best = mops;
        }
        curve[i] = best; dvals[i] = ds[i];
    }
    *nd = n;
    /* granularity = smallest d such that every d' >= d runs at >= 70 % of
       the widest separation (a lone fast point below a slow one, e.g. two
       counters in one 8-byte word handled by store forwarding, is ignored) */
    double ref = curve[n - 1];
    *gran = ds[n - 1];
    for (int i = n - 1; i >= 0 && curve[i] >= 0.7 * ref; i--) *gran = ds[i];
    /* No slowdown even with both counters in one line: the two threads did
       not run concurrently on distinct cores (VM, SMT siblings, scheduler);
       the experiment says nothing.                                         */
    if (*gran == ds[0]) *gran = -1;
    munmap(buf, 1 << 16);
}

/* ── 2. L1 ways / way size ───────────────────────────────────────────── */
static void measure_l1(long *ways_out, long *way_bytes_out, double *conf,
                       int kstar_out[], int *ns_out) {
    const int KMAX = 40;
    size_t region = (size_t)KMAX << 18;               /* up to stride 256 KiB */
    char *buf = big_alloc(region + 4096);
    char *base = (char *)(((uintptr_t)buf + 4095) & ~(uintptr_t)4095);
    long loads = QUICK ? 2000000 : 8000000;
    int kstar[10]; int ns = 0;
    char *addrs[64];
    for (int j = 10; j <= 18 && ns < 10; j++, ns++) {     /* S = 1 KiB .. 256 KiB */
        size_t S = (size_t)1 << j;
        double lat1 = 0; kstar[ns] = KMAX + 1;
        for (int k = 1; k <= KMAX; k++) {
            if ((size_t)k * S > region) break;
            for (int i = 0; i < k; i++) addrs[i] = base + (size_t)i * S;
            link_cycle(addrs, k);
            double lat = chase_best(addrs[0], loads, 3);
            if (k == 1) lat1 = lat;
            else if (lat > 1.5 * lat1) { kstar[ns] = k; break; }
        }
    }
    /* The first stride S at which k* stops falling (k*(S) == k*(2S)) is the
       L1 way size W, and ways = k*(S) - 1.  At still larger strides other
       set-associative structures (the dTLB, indexed by page number) can
       start to conflict and lower k* again, so the *first* plateau is used. */
    long ways = -1, W = -1; int plen = 0;
    for (int i = 0; i + 1 < ns; i++)
        if (kstar[i] <= KMAX && kstar[i] == kstar[i + 1]) {
            ways = kstar[i] - 1; W = 1L << (10 + i);
            plen = 1;
            while (i + plen < ns && kstar[i + plen] == kstar[i]) plen++;
            break;
        }
    /* Sanity: before the plateau k* must not rise again with the stride
       (it does when the L1 index is not a function of the virtual address,
       e.g. with 4 KiB guest pages over a 16 KiB-indexed L1 in a VM), and a
       direct-mapped L1 is implausible.  Either way: report nothing.      */
    if (ways >= 0) {
        int ip = 0; while ((1L << (10 + ip)) != W) ip++;
        for (int i = 0; i < ip; i++) if (kstar[i + 1] > kstar[i]) ways = -1;
        if (ways < 2) ways = -1;
        if (ways < 0) W = -1;
    }
    *ways_out = ways; *way_bytes_out = W;
    *conf = ways < 0 ? 0.0 : (plen >= 3 ? 1.0 : 0.7);
    for (int i = 0; i < ns; i++) kstar_out[i] = kstar[i];
    *ns_out = ns;
    munmap(buf, region + 4096);
}

/* ── 3. L2 ways by minimal thrashing set ─────────────────────────────── */
static char *l2_pool[16384];
static double lat_l2_hit, lat_l2_miss;

static double set_latency(char **set, int k, long loads) {
    char *tmp[16384];
    memcpy(tmp, set, (size_t)k * sizeof *tmp);
    link_cycle(tmp, k);
    return chase_best(tmp[0], loads, 2);
}
/* A set "thrashes" (in L2) if chasing it is slower than chasing the same
   pages at *spread* line offsets by more than
       (L2 hit - L1 hit) + (below-L2 - L2 hit) / 2.
   Same pages means the same dTLB behaviour, so the comparison isolates
   the cache.  The first term allows for the L1 conflicts every
   same-offset set has (all candidates share an L1 set); the second is
   half the cost of an L2 miss.                                          */
static double lat_l1_hit, thrash_margin;
static int thrashes(char **set, int k) {
    if (k < 2) return 0;
    long loads = (long)k * (QUICK ? 20 : 50);
    if (loads < 100000) loads = 100000;
    double lat = set_latency(set, k, loads);
    static char *spread[16384];
    long pg = page_bytes, lines = pg / 64;
    for (int i = 0; i < k; i++) {
        uintptr_t a = (uintptr_t)set[i], base = a & ~(uintptr_t)(pg - 1);
        long off = (long)((a - base) / 64);
        long o2 = (off + 1 + (long)((uint64_t)i * 67 % (uint64_t)(lines - 1))) % lines;
        spread[i] = (char *)(base + (uintptr_t)o2 * 64);
    }
    double ref = set_latency(spread, k, loads);
    return lat - ref > thrash_margin;
}

static int reduce_set(char **S, int n) {
    /* group testing with up to 34 groups, then a linear pass */
    /* Adaptive group testing: try to drop one of g groups; start with
       halves and refine (g doubles, up to 64 > ways + 1 for any cache we
       expect) whenever no group can be dropped.                         */
    int g = 2;
    for (;;) {
        if (n <= g || n <= 2) break;
        int removed = 0;
        for (int gi = 0; gi < g && !removed; gi++) {
            int lo = (int)((long)n * gi / g), hi = (int)((long)n * (gi + 1) / g);
            char *tmp[16384]; int m = 0;
            for (int i = 0; i < n; i++) if (i < lo || i >= hi) tmp[m++] = S[i];
            if (thrashes(tmp, m)) { memcpy(S, tmp, (size_t)m * sizeof *S); n = m; removed = 1; }
        }
        if (!removed) { if (g >= 64) break; g *= 2; }
    }
    for (int i = 0; i < n; ) {
        char *tmp[16384]; int m = 0;
        for (int j = 0; j < n; j++) if (j != i) tmp[m++] = S[j];
        if (thrashes(tmp, m)) { memcpy(S, tmp, (size_t)m * sizeof *S); n = m; }
        else i++;
    }
    return n;
}

static void measure_l2(long l1_bytes, long l2_bytes, long *ways_out, double *conf,
                       int *ntargets, int sizes[], double *hit, double *miss) {
    long pg = page_bytes;
    /* Calibrate: L2-hit latency = random chase over 4 x L1 (fits L2);
       below-L2 latency = chase over 8 x L2.                              */
    {
        size_t a = (size_t)(4 * l1_bytes), b = (size_t)(2 * l2_bytes);
        if (b > ((size_t)1 << 30)) b = (size_t)1 << 30;
        char *buf = big_alloc(b);
        int k = (int)(a / 64); char **ad = malloc((size_t)(b / 64) * sizeof *ad);
        for (int i = 0; i < k; i++) ad[i] = buf + (size_t)i * 64;
        link_cycle(ad, k); lat_l2_hit = chase_best(ad[0], 2000000, 3);
        k = (int)(b / 64);
        for (int i = 0; i < k; i++) ad[i] = buf + (size_t)i * 64;
        link_cycle(ad, k); lat_l2_miss = chase_best(ad[0], 2000000, 3);
        ad[0] = buf; *(char **)buf = buf; lat_l1_hit = chase_best(buf, 2000000, 3);
        free(ad); munmap(buf, b);
    }
    *hit = lat_l2_hit; *miss = lat_l2_miss;
    thrash_margin = (lat_l2_hit - lat_l1_hit) + 0.5 * (lat_l2_miss - lat_l2_hit);
    if (!(lat_l2_miss > 2.0 * lat_l2_hit)) { *ways_out = -1; *conf = 0; *ntargets = 0; return; }

    int N = QUICK ? 4096 : 8192;
    size_t bytes = (size_t)N * pg;
    if (bytes > ((size_t)1 << 30)) { N = (int)(((size_t)1 << 30) / pg); bytes = (size_t)N * pg; }
    char *buf = big_alloc(bytes + pg);
    int T = QUICK ? 5 : 11, got = 0;
    for (int t = 0; t < T; t++) {
        long off = (long)((rnd() % (uint64_t)(pg / 64)) * 64);  /* page offset */
        int n = 0;
        for (int i = 0; i < N; i++) l2_pool[n++] = buf + (size_t)i * pg + off;
        if (!thrashes(l2_pool, n)) continue;
        int m = reduce_set(l2_pool, n);
        sizes[got++] = m;
    }
    *ntargets = got;
    munmap(buf, bytes + pg);
    if (!got) { *ways_out = -1; *conf = 0; return; }
    int best = -1, bestc = 0;
    for (int i = 0; i < got; i++) {
        int c = 0; for (int j = 0; j < got; j++) c += sizes[j] == sizes[i];
        if (c > bestc) { bestc = c; best = sizes[i]; }
    }
    *ways_out = best - 1; *conf = (double)bestc / got;
    /* A minimal thrashing set much larger than any plausible associativity
       means the experiment did not isolate one L2 set (hashed or adaptive
       indexing, non-random physical pages, or a VM): report nothing.      */
    if (best > 65) { *ways_out = -1; *conf = 0; }
}

/* ── 4. latency curve ────────────────────────────────────────────────── */
static int measure_latency(size_t max_bytes, size_t sz[], double lat[]) {
    int n = 0;
    char *buf = big_alloc(max_bytes);
    size_t nl = max_bytes / 64;
    char **ad = malloc(nl * sizeof *ad);
    for (size_t s = (size_t)16 << 10; s <= max_bytes; s *= 2) {
        int k = (int)(s / 64);
        /* one line per 64 B, random order over the whole working set */
        for (int i = 0; i < k; i++) ad[i] = buf + (size_t)i * 64;
        link_cycle(ad, k);
        long loads = QUICK ? 1000000 : 4000000;
        sz[n] = s; lat[n] = chase_best(ad[0], loads, 2); n++;
    }
    free(ad); munmap(buf, max_bytes);
    return n;
}

/* ── 5. dTLB reach ───────────────────────────────────────────────────── */
static int measure_dtlb(int maxpages, int pages[], double lat[]) {
    long pg = page_bytes; int n = 0;
    char *buf = big_alloc((size_t)maxpages * pg);
    char **ad = malloc((size_t)maxpages * sizeof *ad);
    for (int k = 8; k <= maxpages; k *= 2) {
        /* vary the line inside each page so the lines spread over cache sets */
        for (int i = 0; i < k; i++)
            ad[i] = buf + (size_t)i * pg + (size_t)((i * 67) % (pg / 64)) * 64;
        link_cycle(ad, k);
        pages[n] = k; lat[n] = chase_best(ad[0], QUICK ? 1000000 : 3000000, 2); n++;
    }
    free(ad); munmap(buf, (size_t)maxpages * pg);
    return n;
}

static long dczva_bytes(void) {
#if defined(__aarch64__)
    uint64_t v; __asm__ volatile("mrs %0, dczid_el0" : "=r"(v));
    if (v & 16) return 0;                       /* DZP: prohibited */
    return 4L << (v & 15);
#else
    return -1;
#endif
}

static long ram_bytes(void) {
#ifdef __APPLE__
    return sysctl_long("hw.memsize");
#else
    long p = sysconf(_SC_PHYS_PAGES), s = sysconf(_SC_PAGESIZE);
    return p > 0 && s > 0 ? p * s : (8L << 30);
#endif
}

static void print_lvl(const char *name, CacheLvl *c) {
    printf("    %s: {size: %ld, line: %ld, ways: %s%ld, sets: %s%ld, shared_by: %ld}\n",
           name, c->size, c->line, "", c->ways, "", c->sets, c->shared_by);
}
static void print_knees(const char *name, int n, const long xs[], const double ys[]) {
    printf("    %s: [", name); int first = 1;
    for (int i = 1; i < n; i++)
        if (ys[i] > 1.5 * ys[i - 1]) { printf("%s%ld", first ? "" : ", ", xs[i - 1]); first = 0; }
    printf("]\n");
}

int main(int argc, char **argv) {
    int skip_l2 = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) QUICK = 1;
        else if (!strcmp(argv[i], "--skip-l2")) skip_l2 = 1;
        else { fprintf(stderr, "usage: %s [--quick] [--skip-l2]\n", argv[0]); return 2; }
    }
    page_bytes = sysconf(_SC_PAGESIZE);
    ncpu_logical = sysconf(_SC_NPROCESSORS_ONLN);
    os_geometry();
    pin_to(0);

    printf("cacheprobe:\n  version: 1\n");
#ifdef __APPLE__
    printf("  platform: darwin\n");
#elif defined(__linux__)
    printf("  platform: linux\n");
#else
    printf("  platform: other\n");
#endif
    printf("  page_bytes: %ld\n  logical_cpus: %ld\n", page_bytes, ncpu_logical);
    printf("  os:\n");
    print_lvl("L1d", &os_l[1]); print_lvl("L2", &os_l[2]); print_lvl("L3", &os_l[3]);
    fflush(stdout);

    printf("  measured:\n");
    int other = first_non_sibling_cpu();
    int gran, ds[10], nd; double curve[10];
    measure_coherence(other, &gran, curve, ds, &nd);
    if (gran > 0) printf("    coherence_bytes: %d\n", gran); else printf("    coherence_bytes: null\n");
    printf("    coherence_cpus: [0, %d]\n    coherence_mops: {", other);
    for (int i = 0; i < nd; i++) printf("%s%d: %.1f", i ? ", " : "", ds[i], curve[i]);
    printf("}\n"); fflush(stdout);

    long l1w, l1W; double l1c; int ks[10], nks = 0;
    measure_l1(&l1w, &l1W, &l1c, ks, &nks);
    long line = line_os > 0 ? line_os : 64;
    if (l1w > 0)
        printf("    L1d: {ways: %ld, way_bytes: %ld, sets: %ld, confidence: %.2f, kstar_by_stride: {",
               l1w, l1W, (l1W > 0 && line > 0) ? l1W / line : -1L, l1c);
    else
        printf("    L1d: {ways: null, confidence: 0.00, kstar_by_stride: {");
    for (int i = 0; i < nks; i++) printf("%s%ld: %d", i ? ", " : "", 1L << (10 + i), ks[i]);
    printf("}}\n");
    fflush(stdout);

    long l1b = os_l[1].size > 0 ? os_l[1].size : (l1w > 0 && l1W > 0 ? l1w * l1W : 32768);
    long l2b = os_l[2].size > 0 ? os_l[2].size : (1L << 20);
    if (!skip_l2) {
        long l2w; double l2c, hit, miss; int nt, sizes[16];
        measure_l2(l1b, l2b, &l2w, &l2c, &nt, sizes, &hit, &miss);
        if (l2w > 0) printf("    L2: {ways: %ld, ", l2w); else printf("    L2: {ways: null, ");
        printf("confidence: %.2f, targets: %d, hit_ns: %.2f, below_ns: %.2f, set_sizes: [",
               l2c, nt, hit, miss);
        for (int i = 0; i < nt; i++) printf("%s%d", i ? ", " : "", sizes[i]);
        printf("]}\n"); fflush(stdout);
    }

    size_t maxb = (size_t)1 << 30;
    long ram = ram_bytes();
    if (ram > 0 && (size_t)ram / 8 < maxb) maxb = (size_t)ram / 8;
    if (QUICK && maxb > ((size_t)256 << 20)) maxb = (size_t)256 << 20;
    size_t sz[40]; double lat[40]; long szl[40];
    int n = measure_latency(maxb, sz, lat);
    printf("    latency_ns: {");
    for (int i = 0; i < n; i++) { printf("%s%zu: %.2f", i ? ", " : "", sz[i], lat[i]); szl[i] = (long)sz[i]; }
    printf("}\n");
    print_knees("latency_knees_bytes", n, szl, lat);
    fflush(stdout);

    int pages[40]; double tl[40]; long pl[40];
    int np = measure_dtlb(QUICK ? 4096 : 16384, pages, tl);
    printf("    dtlb_ns: {");
    for (int i = 0; i < np; i++) { printf("%s%d: %.2f", i ? ", " : "", pages[i], tl[i]); pl[i] = pages[i]; }
    printf("}\n");
    print_knees("dtlb_knees_pages", np, pl, tl);
    long z = dczva_bytes();
    if (z >= 0) printf("    dczva_bytes: %ld\n", z); else printf("    dczva_bytes: null\n");
    return 0;
}
