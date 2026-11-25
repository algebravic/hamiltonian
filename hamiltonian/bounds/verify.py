import random, sys
from .lift import issq, adj
from .base import find_path, reroutes, max_disjoint
sys.setrecursionlimit(100000)
GLUE=[1, 15, 65, 35, 85, 36, 86, 14, 64, 80, 30, 70, 20, 5, 11, 25, 75, 6, 19, 69, 12, 24, 74, 7, 29, 79, 2, 34, 84, 37, 87, 82, 32, 4, 21, 71, 10, 26, 76, 68, 18, 31, 81, 63, 13, 23, 73, 8, 28, 78, 66, 16, 9, 72, 22, 27, 77, 67, 17, 83, 33, 3]
def lift(b):
    n=len(b); out=[]; segs={}
    for c in range(-12,13): segs[25+c]=[25*b[p]+(1 if p%2==0 else -1)*c for p in range(n)]  # 25+c ... 75+c
    i=0
    while i<len(GLUE):
        v=GLUE[i]
        if v<=12: out.append(v); i+=1
        else:
            w=GLUE[i+1]; s=segs[v] if v in segs else segs[w][::-1]
            assert s[0]==v and s[-1]==w; out+=s; i+=2
    return out
def check_ham(P,N):
    assert sorted(P)==list(range(1,N+1)); assert all(issq(P[i]+P[i+1]) for i in range(N-1)); assert P[0]==1 and P[-1]==3
def check_move(P,i,alt):
    # alt replaces P[i:i+len(alt)], same set, same parity, valid edges
    j=i+len(alt)-1; W=P[i:j+1]
    assert sorted(W)==sorted(alt) and W!=alt
    par={P[p]:p%2 for p in range(i,j+1)}; assert all(par[alt[q]]==(i+q)%2 for q in range(len(alt)))
    Q=[P[i-1]]+alt+[P[j+1]]; assert all(issq(Q[k]+Q[k+1]) for k in range(len(Q)-1))
def alt_for(P,i,j,A):
    S=set(P[i:j+1]); par={P[p]:p%2 for p in range(i,j+1)}; b=P[j+1]; L=j-i+1; orig=P[i:j+1]
    def rec(v,seq,used):
        if len(seq)==L: return list(seq) if issq(v+b) and seq!=orig else None
        for u in A[v]:
            if u in S and u not in used and par[u]==(i+len(seq))%2:
                r=rec(u,seq+[u],used|{u})
                if r: return r
    return rec(P[i-1],[],frozenset())
random.seed(1); n0=43; A=adj(n0)
while True:
    P=find_path(n0,1,3,A); segs=reroutes(P,A)
    if max_disjoint(segs)>=1: break
i,j=segs[0]; alt=alt_for(P,i,j,A)
print("base n0=43 path:",P); print("move window",(i,j),P[i-1:j+2],"->",[P[i-1]]+alt+[P[j+1]])
moves=[(P,i,alt)]
cur=P; curmoves=[(i,alt)]
for level in range(1,3):
    n=len(cur); L=lift(cur); N=25*n+12; check_ham(L,N)
    # locate lifted moves: for each c, lifted window
    newmoves=[]
    pos={v:k for k,v in enumerate(L)}
    for (i,alt) in curmoves:
        for c in range(-12,13):
            sg=lambda p: (1 if p%2==0 else -1)*c
            W=[25*cur[p]+sg(p) for p in range(i,i+len(alt))]
            # position of the window in L (may be reversed)
            k=pos[W[0]]; 
            altL=[25*alt[q]+sg(i+q) for q in range(len(alt))]
            if L[k:k+len(W)]==W: check_move(L,k,altL); newmoves.append((k,altL))
            else:
                k2=pos[W[-1]]; assert L[k2:k2+len(W)]==W[::-1]; check_move(L,k2,altL[::-1]); newmoves.append((k2,altL[::-1]))
    ks=sorted(newmoves); assert all(ks[t+1][0]>=ks[t][0]+len(ks[t][1])+1 for t in range(len(ks)-1))
    print(f"level {level}: N={N} Ham path 1->3 verified; {len(newmoves)} disjoint parity-preserving moves verified")
    cur,curmoves=L,newmoves
