import random, sys, math
from .lift import issq, adj
sys.setrecursionlimit(100000)
def find_path(n,s,t,A,tries=200,budget=200000):
    for _ in range(tries):
        path=[s]; used={s}; steps=[0]
        def rec(v):
            steps[0]+=1
            if steps[0]>budget: return False
            if len(path)==n: return v==t
            cand=[u for u in A[v] if u not in used and (u!=t or len(path)==n-1)]
            random.shuffle(cand)
            cand.sort(key=lambda u: sum(1 for w in A[u] if w not in used))
            for u in cand:
                used.add(u); path.append(u)
                if rec(u): return True
                path.pop(); used.discard(u)
            return False
        if rec(s): return path
    return None
def reroutes(P,A,W=12):
    """windows [i..j] interior (1<=i, j<=n-2); alternative path v_{i-1} -> v_{j+1} on same vertex set,
       each vertex keeping its position parity, different from original."""
    n=len(P); res=[]
    for i in range(1,n-1):
        for j in range(i+1,min(n-1,i+W)):
            S=set(P[i:j+1]); par={P[p]:p%2 for p in range(i,j+1)}
            a,b=P[i-1],P[j+1]; L=j-i+1
            orig=P[i:j+1]; found=[False]
            seq=[]
            def rec(v,used):
                if found[0]: return
                if len(seq)==L:
                    if issq(v+b) and seq!=orig: found[0]=True
                    return
                pos=i+len(seq)
                for u in A[v]:
                    if u in S and u not in used and par[u]==pos%2:
                        used.add(u); seq.append(u); rec(u,used); seq.pop(); used.discard(u)
            rec(a,set())
            if found[0]: res.append((i,j)); break   # shortest window starting at i
    return res
def max_disjoint(segs):
    segs=sorted(segs,key=lambda s:s[1]); cnt=0; last=-10
    for i,j in segs:
        if i>=last+2: cnt+=1; last=j
    return cnt
if __name__=="__main__":
    random.seed(1)
    for n in range(int(sys.argv[1]),int(sys.argv[2]),2):
        A=adj(n); best=None
        for rep in range(int(sys.argv[3])):
            P=find_path(n,1,3,A)
            if P is None: continue
            k=max_disjoint(reroutes(P,A))
            if best is None or k>best[0]: best=(k,P)
        print(n, None if best is None else best[0], flush=True)
