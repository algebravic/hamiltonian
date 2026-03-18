import networkx as nx

def networkx_to_strict_graphbase(
    G,
    filename,
    util_types="ZZZZZZZZZZZZZZZZ",
    default_name="G"
):
    """
    Export a NetworkX graph to a strict Stanford GraphBase file,
    readable by gb_load.
    """

    if len(util_types) != 16:
        raise ValueError("util_types must be exactly 16 characters")

    directed = G.is_directed()

    # Vertex order defines V-line numbers
    vertices = list(G.nodes())
    v_index = {v: i + 1 for i, v in enumerate(vertices)}

    # Graph name (must be quoted)
    graph_name = G.name if getattr(G, "name", "") else default_name

    # Adjacency lists as linked arcs
    arcs = []  # each entry: (to_vertex, next_arc_index)
    first_arc = {v: None for v in vertices}

    for u, v in G.edges():
        # forward arc u -> v
        a = len(arcs) + 1
        arcs.append((v, first_arc[u]))   # next_arc is old head (or 0)
        first_arc[u] = a

        if not directed:
            # reverse arc v -> u
            a = len(arcs) + 1
            arcs.append((u, first_arc[v]))
            first_arc[v] = a

    nV = len(vertices)
    mA = len(arcs)

    with open(filename, "w") as f:
        # Header
        f.write(
            f"* GraphBase graph (util_types {util_types},{nV}V,{mA}A)\n\n"
        )

        # Graph name
        f.write(f"\"{graph_name}\"\n\n")

        # Vertices section
        f.write("* Vertices\n")
        for v in vertices:
            first = first_arc[v]
            label = 'A' + str(first) if first is not None else 0
            f.write(f"\"{v}\",{label}\n")

        # Arcs section
        f.write("\n* Arcs\n")
        for to_v, next_a in arcs:
            if next_a == 0:
                f.write(f"V{v_index[to_v]},0\n")
            else:
                f.write(f"V{v_index[to_v]},A{next_a}\n")
