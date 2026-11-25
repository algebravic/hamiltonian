import sys, resource, time
sys.path.insert(0,"kobayashi-pw/src"); sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from wmodel import beta_without, make_model, graph
from kobayashi_pw.orders import OrderDAG, optimize, make_model_v7
n, which = int(sys.argv[1]), sys.argv[2]
dag = OrderDAG.load(f"sq/sq{n}.dag"); dag.g = graph(n)
narcs = sum(len(a) for a in dag.arcs.values())
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
model = make_model_v7() if which == "v7" else make_model(beta_without(n), n)
t = time.time(); r = optimize(dag, model); dt = time.time() - t
r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(f"n={n} {which}: nodes={len(dag)} arcs={narcs} RSS after load={r0:.0f} MB, after optimise={r1:.0f} MB, max_labels={r.max_labels}, {dt:.0f}s")
