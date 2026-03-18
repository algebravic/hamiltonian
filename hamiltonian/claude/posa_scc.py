"""
posa_scc.py  —  Strongly connected components of the directed Posa rotation graph.

Node = canonical Hamiltonian path (first endpoint ≤ last endpoint).
Directed edge P → Q if Q is obtainable from P by a single Posa rotation
from either endpoint of P (reversals are free → both endpoints available).

Minimum basis size = number of SCCs with in-degree 0 in the condensation DAG
(sources of the condensation = SCCs that no other SCC can reach).

A path is NOT a valid basis element iff its SCC has in-degree ≥ 1 in the
condensation (i.e., it is already reachable from somewhere else).
"""

import ctypes, subprocess, hashlib, os, sys
from math import isqrt
from collections import defaultdict

# -----------------------------------------------------------------------
# C path enumerator (same as posa_orbits.py)
# -----------------------------------------------------------------------
_C_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define MAXN 64
static int adj[MAXN][MAXN],deg[MAXN],path[MAXN],vis[MAXN],N;
static int32_t*buf=NULL; static size_t cap=0,cnt=0;
static void grow(){if(cnt>=cap){cap=cap?cap*2:65536;buf=realloc(buf,cap*N*4);}}
static void dfs(int d){
    if(d==N){grow();int32_t*dst=buf+cnt*N;
        if(path[0]<path[N-1]){for(int i=0;i<N;i++)dst[i]=path[i];}
        else{for(int i=0;i<N;i++)dst[i]=path[N-1-i];}cnt++;return;}
    int v=path[d-1];
    for(int k=0;k<deg[v];k++){int w=adj[v][k];if(!vis[w]){vis[w]=1;path[d]=w;dfs(d+1);vis[w]=0;}}
}
int32_t*enumerate_paths(int n,const int*edges,int ne,size_t*out){
    N=n;memset(deg,0,sizeof(deg));memset(adj,0,sizeof(adj));
    for(int i=0;i<ne;i++){int u=edges[2*i],v=edges[2*i+1];adj[u][deg[u]++]=v;adj[v][deg[v]++]=u;}
    buf=NULL;cap=0;cnt=0;
    for(int s=1;s<=n;s++){memset(vis,0,sizeof(vis));vis[s]=1;path[0]=s;dfs(1);}
    *out=cnt;return buf;
}
void free_paths(int32_t*p){free(p);}
"""
_LIB = None
def _get_lib():
    global _LIB
    if _LIB: return _LIB
    h = hashlib.md5(_C_SRC.encode()).hexdigest()[:8]
    so = f'/tmp/posa_scc_{h}.so'
    if not os.path.exists(so):
        src = f'/tmp/posa_scc_{h}.c'
        open(src,'w').write(_C_SRC)
        subprocess.check_call(['gcc','-O3','-o',so,'-shared','-fPIC',src])
    lib = ctypes.CDLL(so)
    lib.enumerate_paths.restype  = ctypes.POINTER(ctypes.c_int32)
    lib.enumerate_paths.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_int),
                                     ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]
    lib.free_paths.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    _LIB = lib; return lib

def get_paths(n, adj_dict):
    lib = _get_lib()
    edges=[x for u in range(1,n+1) for v in adj_dict[u] if u<v for x in(u,v)]
    ne=len(edges)//2
    c_e=(ctypes.c_int*len(edges))(*edges)
    np_=ctypes.c_size_t(0)
    ptr=lib.enumerate_paths(n,c_e,ne,ctypes.byref(np_))
    raw=[ptr[i] for i in range(np_.value*n)]
    lib.free_paths(ptr)
    seen=set();result=[]
    for k in range(np_.value):
        t=tuple(raw[k*n:(k+1)*n])
        if t not in seen: seen.add(t);result.append(t)
    return result

# -----------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------
def build_adj(n):
    adj={v:set() for v in range(1,n+1)}
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            s=i+j; r=isqrt(s)
            if r*r==s: adj[i].add(j);adj[j].add(i)
    return adj

def posa_successors(path, adj):
    """All paths reachable by one Posa rotation from either endpoint."""
    p=path; n=len(p)
    if n<4: return []
    result=[]
    # Right endpoint
    right=p[-1]
    for i in range(1,n-2):
        if right in adj[p[i]]:
            q=p[:i+1]+tuple(reversed(p[i+1:]))
            result.append(q if q[0]<q[-1] else tuple(reversed(q)))
    # Left endpoint (reverse and rotate)
    rev=tuple(reversed(p)); left=rev[-1]
    for i in range(1,n-2):
        if left in adj[rev[i]]:
            q=rev[:i+1]+tuple(reversed(rev[i+1:]))
            result.append(q if q[0]<q[-1] else tuple(reversed(q)))
    return result

# -----------------------------------------------------------------------
# Tarjan's SCC (iterative)
# -----------------------------------------------------------------------
def tarjan_scc(n_nodes, out_edges):
    """
    Iterative Tarjan's algorithm.
    Returns list of SCCs, each a list of node indices.
    """
    index_counter=[0]
    stack=[]
    lowlink=[0]*n_nodes
    index=[-1]*n_nodes
    on_stack=[False]*n_nodes
    sccs=[]

    def strongconnect(v0):
        # Iterative version using explicit stack
        call_stack=[(v0,iter(out_edges[v0]),False)]
        while call_stack:
            v,it,returning=call_stack[-1]
            if not returning:
                # First visit
                index[v]=lowlink[v]=index_counter[0]
                index_counter[0]+=1
                stack.append(v); on_stack[v]=True
                call_stack[-1]=(v,it,True)
            try:
                w=next(it)
                if index[w]==-1:
                    call_stack.append((w,iter(out_edges[w]),False))
                elif on_stack[w]:
                    lowlink[v]=min(lowlink[v],index[w])
            except StopIteration:
                call_stack.pop()
                if call_stack:
                    parent=call_stack[-1][0]
                    lowlink[parent]=min(lowlink[parent],lowlink[v])
                # Root of SCC
                if lowlink[v]==index[v]:
                    scc=[]
                    while True:
                        w=stack.pop(); on_stack[w]=False; scc.append(w)
                        if w==v: break
                    sccs.append(scc)

    for v in range(n_nodes):
        if index[v]==-1:
            strongconnect(v)
    return sccs


# -----------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------
def analyse(n, verbose=False):
    import time
    adj   = build_adj(n)
    t0    = time.time()
    paths = get_paths(n, adj)
    t_enum = time.time()-t0
    np_   = len(paths)
    if np_==0:
        return dict(n=n, n_paths=0, n_scc=0, n_sources=0,
                    source_sizes=[], non_source_sizes=[], all_sizes=[])

    path_id={p:i for i,p in enumerate(paths)}

    # Build directed edge lists
    t0=time.time()
    out_edges=[[] for _ in range(np_)]
    for pid,p in enumerate(paths):
        for q in posa_successors(p, adj):
            qid=path_id.get(q)
            if qid is not None and qid!=pid:
                out_edges[pid].append(qid)
    t_build=time.time()-t0

    # Tarjan SCC
    t0=time.time()
    # Fix the -1 initialisation (Python uses - not -)
    index=[-1]*np_; lowlink=[0]*np_; on_stack=[False]*np_
    index_counter=[0]; stack=[]; sccs=[]
    def sc(v0):
        cs=[(v0,iter(out_edges[v0]))]
        while cs:
            v,it=cs[-1]
            if index[v]==-1:
                index[v]=lowlink[v]=index_counter[0]
                index_counter[0]+=1
                stack.append(v); on_stack[v]=True
            try:
                w=next(it)
                if index[w]==-1:
                    cs.append((w,iter(out_edges[w])))
                elif on_stack[w]:
                    lowlink[v]=min(lowlink[v],index[w])
            except StopIteration:
                cs.pop()
                if cs:
                    lowlink[cs[-1][0]]=min(lowlink[cs[-1][0]],lowlink[v])
                if lowlink[v]==index[v]:
                    scc=[]
                    while True:
                        w=stack.pop();on_stack[w]=False;scc.append(w)
                        if w==v: break
                    sccs.append(scc)
    for v in range(np_):
        if index[v]==-1: sc(v)
    t_scc=time.time()-t0

    # Condensation: find source SCCs (in-degree 0 in condensation)
    node_scc={}
    for sid,scc in enumerate(sccs):
        for v in scc: node_scc[v]=sid

    scc_in_deg=[0]*len(sccs)
    for pid in range(np_):
        sid=node_scc[pid]
        for qid in out_edges[pid]:
            sqid=node_scc[qid]
            if sqid!=sid:
                scc_in_deg[sqid]+=1

    sources=[i for i,d in enumerate(scc_in_deg) if d==0]
    non_sources=[i for i,d in enumerate(scc_in_deg) if d>0]

    source_sizes=sorted(len(sccs[i]) for i in sources)
    non_source_sizes=sorted(len(sccs[i]) for i in non_sources)
    all_sizes=sorted(len(s) for s in sccs)

    # For each source SCC, look at representative paths for patterns
    source_paths=[]
    for sid in sources:
        for v in sccs[sid]:
            source_paths.append(paths[v])

    if verbose:
        print(f"  enum={t_enum:.2f}s  build={t_build:.2f}s  scc={t_scc:.3f}s")
        print(f"  {np_} paths, {len(sccs)} SCCs")
        print(f"  Source SCCs ({len(sources)}): sizes={source_sizes}")
        print(f"  Non-source SCCs ({len(non_sources)}): sizes={non_source_sizes}")

        # Analyse endpoint pairs of source vs non-source paths
        from collections import Counter
        def ep_type(p):
            a,b=p[0],p[-1]
            s=a+b
            r=isqrt(s)
            return ('sqsum' if r*r==s else 'nonsq'), (a,b)

        src_ep  = Counter(ep_type(p)[0] for p in source_paths)
        # non-source paths
        non_src_pids=[v for sid in non_sources for v in sccs[sid]]
        non_src_paths=[paths[v] for v in non_src_pids]
        nsrc_ep = Counter(ep_type(p)[0] for p in non_src_paths)
        print(f"  Source path endpoints sum to square: {src_ep}")
        print(f"  Non-source path endpoints sum to square: {nsrc_ep}")

    return dict(n=n, n_paths=np_, n_scc=len(sccs),
                n_sources=len(sources), n_non_sources=len(non_sources),
                source_sizes=source_sizes, non_source_sizes=non_source_sizes,
                all_sizes=all_sizes,
                paths=paths, sccs=sccs, node_scc=node_scc,
                sources=sources, non_sources=non_sources,
                out_edges=out_edges, path_id=path_id)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("start_n",type=int,nargs="?",default=15)
    p.add_argument("end_n",type=int,nargs="?",default=None)
    p.add_argument("-v","--verbose",action="store_true")
    args=p.parse_args()
    start=args.start_n
    end=args.end_n if args.end_n is not None else start

    print(f"{'n':>4}  {'paths':>8}  {'SCCs':>6}  {'sources':>8}  "
          f"{'src_sizes':>30}  non_src_sizes")
    print("-"*90)
    for nn in range(start,end+1):
        r=analyse(nn,verbose=args.verbose)
        if r['n_paths']==0:
            print(f"{nn:>4}  {'0':>8}  {'0':>6}  {'0':>8}")
            continue
        print(f"{nn:>4}  {r['n_paths']:>8,}  {r['n_scc']:>6,}  "
              f"{r['n_sources']:>8,}  "
              f"{str(r['source_sizes']):>30}  {r['non_source_sizes']}")
