"""
Pool all completed paths across iterations for a combined estimate.
Since Q doesn't change much with IS-weighted update, pooling is approximate.
Also try a cleaner approach: NO CE update, just pool many samples with fixed Q.
"""
import numpy as np
from math import isqrt, log
import time, sys
sys.path.insert(0, '/home/claude')
from ce_sampler import sample_batch

EXACT = {57: 68_092_497_615, 59: 332_409_010_079}

def build_adj(n):
    adj=[[] for _ in range(n+1)]
    for u in range(1,n+1):
        for v in range(u+1,n+1):
            s=u+v; r=isqrt(s)
            if r*r==s: adj[u].append(v); adj[v].append(u)
    return adj

def adj_to_arrays(adj, n):
    offsets=np.zeros(n+2,dtype=np.int32)
    for u in range(1,n+1): offsets[u+1]=offsets[u]+len(adj[u])
    flat=np.zeros(offsets[n+1],dtype=np.int32)
    for u in range(1,n+1): flat[offsets[u]:offsets[u+1]]=adj[u]
    return flat, offsets

def warmup():
    n0=10; adj0=build_adj(n0); fl0,of0=adj_to_arrays(adj0,n0)
    w0=np.ones((n0-1,n0+1,n0+1),dtype=np.float64)
    sample_batch(fl0,of0,w0,n0,5,99)

print("JIT warmup...", end=' ', flush=True); warmup(); print("done")

for n in [57, 59]:
    exact = EXACT[n]
    adj = build_adj(n); flat,offsets=adj_to_arrays(adj,n); n_s=n-1
    
    # Warnsdorff weights (fixed — no CE update)
    w=np.zeros((n_s,n+1,n+1),dtype=np.float64)
    for u in range(1,n+1):
        for v in adj[u]:
            for k in range(n_s):
                w[k,u,v]=1.0/(len(adj[v])+0.1)
    
    print(f"\n{'='*65}")
    print(f"n={n}  |Ham(G_n)| = {exact:.6e}")
    print(f"Strategy: pool many runs with fixed Warnsdorff IS distribution")
    print(f"{'─'*65}")
    
    all_log_w = []
    M_total   = 0
    t_start   = time.perf_counter()
    
    batch_M = 2_000_000
    n_batches = 15  # 15 × 2M = 30M total samples, ~75s
    
    print(f"{'batch':>6}  {'new_done':>8}  {'cumul_done':>10}  "
          f"{'running_est':>12}  {'ratio':>7}  {'CV':>6}")
    
    for b in range(n_batches):
        lqs, pdata, n_done = sample_batch(flat,offsets,w,n,batch_M,b*13+n*7)
        if n_done > 0:
            all_log_w.extend((-lqs[:n_done]).tolist())
        M_total += batch_M
        
        if all_log_w:
            log_w = np.array(all_log_w)
            max_lw= log_w.max()
            log_est=max_lw+log(np.exp(log_w-max_lw).sum())-log(M_total)
            est=float(np.exp(log_est))
            
            w_pos = np.exp(log_w)
            w_all = np.zeros(M_total); w_all[:len(w_pos)]=w_pos
            se   = w_all.std(ddof=1)/np.sqrt(M_total)
            cv   = se/est
            
            print(f"{b:6d}  {n_done:8d}  {len(all_log_w):10d}  "
                  f"{est:12.4e}  {est/exact:7.4f}  {cv:6.3f}")
    
    elapsed = time.perf_counter()-t_start
    # Final pooled estimate
    log_w=np.array(all_log_w)
    max_lw=log_w.max()
    log_est=max_lw+log(np.exp(log_w-max_lw).sum())-log(M_total)
    final_est=float(np.exp(log_est))
    w_pos=np.exp(log_w)
    w_all=np.zeros(M_total); w_all[:len(w_pos)]=w_pos
    se=w_all.std(ddof=1)/np.sqrt(M_total)
    
    print(f"\n{'─'*65}")
    print(f"Total samples:     {M_total:>14,}")
    print(f"Completed paths:   {len(all_log_w):>14,}")
    print(f"Completion rate:   {len(all_log_w)/M_total:>14.4%}")
    print(f"Elapsed:           {elapsed:>14.1f}s")
    print(f"Estimate:          {final_est:>14.4e}")
    print(f"Exact:             {exact:>14.4e}")
    print(f"Ratio:             {final_est/exact:>14.4f}")
    print(f"± 1σ:              {se:>14.2e}  ({se/final_est:.1%})")
    ess=w_pos.sum()**2/(w_pos**2).sum()
    print(f"ESS (within done): {ess:>14.1f} / {len(all_log_w)}")
