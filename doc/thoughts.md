The user wants to implement option 2: a new lighter "stochastic DP" that tracks a particle population for SMC on the DP. This is a substantial implementation task.

Let me think about what this involves:

The DP structure for G_n: We need to understand the pathwidth decomposition and the DP states.
SMC on the DP: We maintain M particles, each representing a frontier state. At each DP step, we extend particles, weight them, and resample when needed.
Let me think about what the DP state looks like for counting Ham paths in G_n.

For the pathwidth decomposition of G_n, vertices are introduced in natural order 1 through n, and at each step the frontier tracks which vertices from the current prefix still connect to the remaining graph. Since the pathwidth is bounded around 14, the frontier size stays manageable. The DP state then needs to encode which vertices in the frontier are part of the Hamiltonian path being constructed. For each frontier vertex, I need to track whether it's free (not yet connected), full (degree 2 in the path segments), or open (an endpoint of a segment). The key constraint is that open vertices pair up as endpoints of path segments, and the configuration must avoid cycles. Looking at the actual C implementation, it uses a frontier-based DP where states map frontier configurations to counts, and for the SMC approach I need to understand this state space and build a particle filter around it.

Rather than working through the pathwidth decomposition directly, I'm thinking about a simpler approach: build Hamiltonian paths step by step in a fixed vertex introduction order (1, 2, ..., n), deciding at each step how the new vertex connects to existing path segments. This is fundamentally different from greedy path construction and requires careful state management.

For a Hamiltonian path on vertices 1..n, after introducing vertices 1..k, the state needs to track which vertices are open endpoints (still able to connect to later vertices) versus closed endpoints (already have degree 2). Since every vertex must be in the path, I need to encode the path segment structure and identify which frontier vertices can still accept connections from {k+1..n}.

The key insight is that the state is really a matching on the frontier vertices—tracking which ones are paired together as parts of the same segment and which are free endpoints. When I introduce vertex k+1, I can either extend existing segments, merge segments, or start a new one.

For the SMC approach, I'll precompute which vertices at each step have edges forward, then represent each particle as a matching on that frontier. At transitions, I update the matching based on the edges k+1 can form, and prune vertices that have no more outgoing edges. The final state should have exactly one segment with two open endpoints to form a valid Hamiltonian path. For the graph G_n where vertices connect if their sum is a perfect square, each vertex u has a "last neighbor" — the highest-numbered vertex it connects to — and once we process past that point, u can leave the frontier. The particle state tracks a partial matching on the frontier vertices, representing which ones are already paired together in the evolving path.

But I'm realizing the state needs to account for isolated vertices too — ones whose edges all lead to future vertices. For a Hamiltonian path, every vertex must eventually connect, so an isolated vertex at step k isn't a dead end; it just hasn't found its connections yet among the vertices we've processed so far.

The DP state should track which frontier vertices are open ends of path components. I can represent this as a matching on the frontier: paired vertices are the two endpoints of the same path, and unmatched vertices are isolated (single-vertex components waiting for future connections). This matching captures the essential structure of how the partial solution on vertices 1..k decomposes into disjoint paths.

When adding vertex k+1, I start it as isolated and decide which frontier vertices to connect it to via edges. The key constraint is that k+1 can have at most 2 neighbors from the frontier, and connecting to a frontier vertex means either extending an existing path or merging two path endpoints into one component.

To avoid cycles, I can't connect two vertices that already belong to the same component unless we're closing out a Hamiltonian path at the very end. This is getting intricate, so I'm thinking about how to represent the state more cleanly—maybe as a tuple indexed by frontier vertices, where each entry tracks whether that vertex is isolated, an endpoint of some component, or already interior to a path. I could use a canonical form where component labels are assigned in order of first appearance to keep the representation minimal.

Rather than trying to parse through thousands of lines of existing C code, let me approach this from scratch. The key observation is that for the square-sum graph on vertices 1 through n, the frontier at step k consists of vertices that have already been introduced but still have edges reaching into the not-yet-introduced vertices. Since each vertex v connects to vertices w where v+w is a perfect square, I can determine exactly when v leaves the frontier—it's when we've processed all its neighbors. The pathwidth being around 14 suggests the frontier stays manageable throughout the algorithm.

I'll represent each particle as a state encoding how the frontier vertices connect via paths. Each frontier vertex gets a label indicating whether it's isolated, fully connected internally, or an endpoint of some path component. When I introduce a new vertex, I need to consider which frontier vertices it connects to, respecting the constraint that path vertices have degree at most 2, and then update the state accordingly.

The SMC approach samples one valid transition per particle at each step rather than enumerating all of them. If a particle has no valid transitions it dies, but if it has k valid transitions I sample one uniformly and weight it by k. This branching factor becomes the key—instead of exploring an exponential tree, I'm sampling paths and reweighting to account for the branches I didn't explore. The argument is that this gives the correct marginal distribution over Hamiltonian paths because the weights compensate for the sampling. At the final step, I check if each particle's path is valid—exactly one component with two endpoints and all constraints satisfied—to count toward the total Hamiltonian paths. For resampling, I'll periodically redistribute particles to reduce variance while preserving the expected weight sum. Now I'm setting up the core data structures: building the adjacency list, precomputing when each vertex leaves the frontier based on its last forward neighbor, and representing particle states as dictionaries tracking which component each vertex belongs to. state_dict: {frontier_vertex: role}

Since the frontier contains at most ~15 vertices, I can represent the state as a tuple indexed by frontier position with role values.

When introducing vertex v at step k, I add it to the frontier as isolated, then enumerate which of its backward neighbors to connect to—accounting for the constraint that v can connect to at most 2 neighbors without creating cycles or exceeding degree limits. The tricky part is handling when vertices leave the frontier: once a vertex u's last forward neighbor is introduced, u must exit, and at that point it needs to be either interior with degree 2 or an endpoint with degree 1—isolated vertices would violate the Hamiltonian path requirement since they'd have no connections at all. u is isolated (degree 0), the current partial path is invalid.

Wait, but what if u has no neighbors in the introduced vertices? Then u is isolated and must connect to future vertices. But if u is leaving the frontier, it has no future connections. Contradiction — this means u has no path connections at all, so the current partial solution doesn't include u in any path.

For a Ham path, every vertex must be in the path. So if u leaves the frontier as isolated, this branch is invalid (contributes 0 to the count).

Simpler observation: For G_n with natural ordering, every vertex has at least one neighbor (otherwise it's isolated and there's no Ham path through it). So I need to check that when a vertex leaves the frontier, it's not isolated.

OK let me now write the actual implementation. I'll keep it clean and well-documented.

Actually, let me look at the graph structure more carefully. For G_n with vertices 1..n, ordered naturally, I need to understand the neighbor relationships. Each vertex v has backward neighbors (vertices less than v that sum with v to a perfect square) and forward neighbors (vertices greater than v with the same property). The frontier at any step contains vertices that still have potential forward connections. When I introduce a new vertex, I need to enumerate which backward neighbors it connects to from the frontier, ensuring no vertex becomes isolated when it leaves. I need to be more precise about what constitutes the frontier at each step. The frontier after introducing vertex k should contain all vertices u ≤ k whose maximum forward neighbor is greater than k, meaning they still have edges to introduce in future steps. A vertex leaves the frontier once all its forward edges have been processed. I need to be more careful about the frontier definition. A vertex u is in the frontier after introducing vertices 1 through k if it's already been introduced (u ≤ k) and still has at least one neighbor that hasn't been introduced yet (some w > k). This means frontier_k = {u ≤ k : last_fwd[u] > k}. When transitioning from step k-1 to step k, I need to figure out which vertices enter and leave the frontier—specifically, vertices with last_fwd[u] = k will leave since they no longer have forward neighbors beyond k. After step k (frontier_k): {u ≤ k : last_fwd[u] > k} = {u ≤ k : last_fwd[u] ≥ k+1}

Vertices ENTERING frontier at step k: {k} if last_fwd[k] > k (i.e., k has a forward neighbor)
Vertices LEAVING frontier at step k: {u ≤ k-1 : last_fwd[u] = k} (their last forward neighbor is exactly k)

And at step k, we process the edge (u, k) for each u ∈ frontier_{k-1} with (u,k) ∈ E (i.e., backward neighbors of k that are in the frontier).

The backward neighbors of k are vertices u < k where u+k is a perfect square. Since all backward neighbors have last_fwd[u] ≥ k, they're guaranteed to be in frontier_{k-1}, so when introducing vertex k I can connect it to any subset of these backward neighbors.

Now for the state representation: I need to track a path matching on the current frontier, where each frontier vertex records how many path edges it's already used—either 0 (isolated, waiting to connect to future vertices), 1 (endpoint of a path segment with room for one more connection), or 2 (fully matched, no more connections possible). I'll represent this as a component label for each frontier vertex, with components forming a canonical partition where each component has up to 2 endpoints and interior vertices marked as full.

For a frontier of size 14, the state space is roughly 3^14 times a Catalan-like factor, which is large in theory but manageable in practice since many states are invalid and the actual frontier size stays much smaller than 14 for natural orderings. The key observation is that the frontier at step k consists of vertices u ≤ k where last_fwd[u] > k, and this set tends to be quite constrained.

Now I'm thinking about the actual implementation. For building a Hamiltonian path on G_n, I can simplify the state significantly: at each step k, the partial solution is a collection of path segments on vertices 1 through k, where each segment has two open ends (or one if it's an isolated vertex). This gives me a cleaner representation for the state space in the search. is:

A matching M on the current frontier vertices: pairs {u, v} where u and v are the two open ends of the same path segment
Isolated vertices: vertices in the frontier that haven't been connected yet (they'll connect to future vertices)
"Full" vertices: frontier vertices with degree 2 (interior) — these are in the frontier because they have future edges, but their path role is fixed
Wait, a vertex with degree 2 is "interior" in the path but might still be in the frontier if it has edges to future vertices. Such edges can't be used (the vertex already has 2 path connections), so when we introduce a new vertex that's adjacent to this interior vertex, we simply don't connect them.

So the state more precisely is:

For each frontier vertex v: role[v] ∈ {isolated, endpoint(i), interior}
isolated: degree 0 in current path (needs future connections)
endpoint(i): degree 1, part of path segment i
interior: degree 2, can't accept more connections
And the constraint: endpoint(i) appears exactly twice for each i (the two ends of segment i).

An isolated frontier vertex with degree 0 could start a new path segment that gets closed later through future connections, which is valid. For the state to be valid, I need no cycles, each frontier vertex with degree at most 2, and non-frontier vertices either fully closed with degree 2 or serving as endpoints — but a non-frontier vertex with degree 1 would be a dangling endpoint that never gets completed, which is impossible.

So when a vertex leaves the frontier, it must have degree at least 1 to be part of the final Hamiltonian path. Now I'm ready to implement this. I'm setting up the class structure and precomputing some helper data like the last forward neighbor for each vertex.

Setting up the implementation... Still writing particle filter logic...

Now I'm thinking through the state representation more carefully—each frontier vertex needs a label to track whether it's isolated, matched, or an endpoint, and I need to figure out how to encode the matching structure efficiently in the particle state. yet connected to any path segment)

-1: interior (degree 2, can't accept more connections)
positive integer j: endpoint of component j (two vertices with the same j are ends of the same path)
For canonical form: labels are assigned in order of first appearance (1, 2, 3, ...).

Example frontier = {a, b, c, d, e}:

State (1, 1, 0, 2, 2) means: a and b are endpoints of segment 1, c is isolated, d and e are endpoints of segment 2
This representation uses a small integer per frontier vertex.

Transition:

When introducing vertex k:

Current frontier = F_{k-1}
Backward neighbors B_k = {w ∈ F_{k-1} : (w,k) ∈ E}
Valid connections: S ⊆ B_k with |S| ≤ 2, avoiding cycles
For each valid connection S:
Create new state where k is connected to vertices in S
Handle vertices leaving the frontier (those with last_fwd[v] = k)
Transition enumeration:

The backward neighbors B_k typically have 2-4 vertices. For each valid connection subset S, I need to check whether k can remain isolated or must connect to something. If k has no future edges (last_fwd[k] = 0), then S = {} would leave k isolated with no way to integrate into the Hamiltonian path, making that transition invalid. So empty connections only work if k will have future connections ahead.

When S = {w}, k connects to exactly one backward neighbor. If k is an endpoint (degree 1), it joins w's component; if w is isolated, they form a new two-vertex path; if w is already an endpoint of some component, k becomes the other endpoint; but if w is interior with degree 2, that's invalid. When S = {w1, w2}, k becomes interior with degree 2, connecting to both backward neighbors—if both are isolated, they merge into a path segment with w1 and w2 as endpoints and k interior.

When one is isolated and the other is an endpoint of an existing segment, k extends that segment by attaching to the endpoint, shifting the structure so the isolated vertex becomes the new endpoint. If both w1 and w2 are endpoints of different segments, k merges them together, creating one longer segment with the two original non-w endpoints as the new endpoints.

However, if both endpoints belong to the same segment, that would create a cycle, which is invalid for a Hamiltonian path. Similarly, connecting to any interior vertex is impossible since those can't accept additional connections.

Now I need to implement this logic. I'll represent the frontier state using a dictionary mapping each vertex to its component label—0 for isolated, -1 for interior, and positive integers for endpoint identifiers—plus a separate dictionary tracking which two vertices are the endpoints of each component to detect cycles.

Actually, given the small frontier size, a tuple-based representation might be cleaner. Let me start coding this carefully.

Writing the transition logic... Continuing with the transition logic...

Now I'm enumerating all valid connection subsets—starting with the empty set if the vertex has future edges, then single connections to each backward neighbor, and finally pairs of connections where both neighbors aren't interior vertices.

I'm realizing this approach is getting complex, so let me simplify by representing state as a mapping where each frontier vertex maps to a component ID: 0 for isolated, -1 for interior, or a positive integer for component endpoints. This way I can maintain the invariant that each positive value appears exactly twice, representing the two endpoints of a component.

When connecting k to a subset S of backward active neighbors, I'll handle each case explicitly: isolated vertices get paired with k under a new component ID, vertices that are already endpoints join their existing component with k as the new endpoint, and if |S| = 2 then k becomes interior. Let me just code this up with clear case handling rather than trying to be too clever about it. we introduce vertex new_v:

new_v enters the frontier with role 0 (will be updated)
Backward neighbors B = [w for w in adj[new_v] if w < new_v and w in frontier]
Enumerate subsets S ⊆ {w ∈ B : state[w] != -1} with |S| ≤ 2
For each S: a. Compute new_state = updated state after connecting new_v to S b. Check validity (no cycles if |S|=2 and both endpoints of same component) c. Close vertices leaving frontier (check they're not isolated if no future edges) d. If valid, add to transitions list
Also: if new_v has no forward edges (last_fwd[new_v] = 0), new_v must have |S| ≥ 1 (otherwise new_v is isolated = not in path = invalid)
Now I'm working through how to compute the new state based on which vertices new_v connects to. When S is empty, new_v stays isolated with role 0. When S contains a single vertex w, I need to distinguish between w being isolated (creating a new component) versus w being an endpoint of an existing component—though I'm realizing my logic for handling the endpoint case needs refinement, since marking w as interior doesn't quite capture the full state transition. Actually when w (endpoint of segment j) connects to new_v:

The segment's endpoint w gets degree 2 (w connects to: its segment neighbor + new_v)
Wait, that means w becomes interior!
new_v becomes the new endpoint of segment j
Actually: segment j has endpoints w and other_end.
When new_v connects to w: the segment extends to: other_end -- ... -- w -- new_v
So w now has degree 2 (interior), new_v has degree 1 (endpoint)
new_state[w] = -1 (interior)
new_state[new_v] = j (new endpoint of segment j)
(other_end still has label j, meaning other_end and new_v are the two endpoints)

Now for the case where I have two neighbors in S: if both w1 and w2 are isolated, I create a new segment connecting them through new_v, marking new_v as interior. If w1 is isolated but w2 is already an endpoint of segment j, I extend that segment to include w1, making new_v interior and updating the endpoint labels accordingly. When two different segments merge through the new vertex, I need to find the other endpoints of each segment, then connect them into one longer path where the new vertex and the two boundary vertices become interior nodes. I'll assign a fresh label to the merged segment's endpoints and mark the interior nodes accordingly.

If both boundary vertices belong to the same component, that creates a cycle, which is invalid for a Hamiltonian path. Now I'm working through what happens when a vertex leaves the frontier—specifically, when its last forward neighbor gets introduced, it can no longer extend forward. If it's isolated at that point, the path is invalid, but if it's already interior with degree 2, that's fine.

When a vertex leaves as an endpoint, it stays as an endpoint of the final path, which is acceptable as long as we end up with exactly 2 endpoints total. Once a vertex closes, I remove it from tracking and update the frontier accordingly.

I'm reconsidering the state representation to make this cleaner—instead of tracking everything in one dict, I'll separate the currently-open frontier vertices from the count of endpoints that have already closed. This way I can track how many endpoints have permanently left the frontier and validate at the end that the total endpoint count (closed plus remaining open) equals exactly 2.

The key insight is that I need to count path components correctly: each component is identified by its distinct positive labels, and I need to ensure the final state has exactly one connected path with both endpoints accounted for. Let me just implement this concretely and test it rather than overthinking the representation.

I'll use a dictionary-based state where each vertex in the frontier maps to its partner in the same component—isolated vertices map to themselves, and interior vertices map to None. This way I can track which endpoints belong together and enumerate valid transitions exhaustively since the graph has small degree and pathwidth.

Implementing the state representation... an endpoint whose partner left the frontier, so it's waiting to potentially reconnect or close. When v is interior (degree 2): v is not in state at all.

This encoding is cleaner—I track only frontier vertices and their relationships, using None to mark when a component has one endpoint closed and one still open.

But I'm realizing the state space is still complex because vertices can reconnect in unexpected ways. Let me step back and think about what I actually need to track for counting Hamiltonian paths: which vertices are currently frontier endpoints, and how they're paired within components.

Implementing the state tracking... I'm settling on a state dictionary that tracks frontier vertices with sentinel values: ISOLATED for degree-0 vertices waiting to connect, INTERIOR for absorbed vertices, None for endpoints whose partners have left the frontier, and vertex references for paired endpoints of the same segment. To detect cycles, I'll check if both endpoints of a segment are backward neighbors of the current vertex, which would create a problematic loop. Now let me write out the full implementation and test it on small cases.

Writing the implementation... Finishing the cycle detection logic...

The key insight is that connecting vertex k to both w1 and w2 creates a cycle only if they're already in the same connected component. My partner-based state tracking handles the direct case—when state[w1] == w2, they're endpoints of the same segment, so connecting them would close a cycle. But when endpoints have left the frontier (marked CLOSED), I lose information about which segments they belong to, making it harder to determine if two CLOSED vertices are actually in the same component or different ones.

I need proper component tracking instead. I'll use integer labels where positive values represent open component endpoints, zero means isolated vertices awaiting assignment, and negative values mark interior vertices. Each component label appears 0, 1, or 2 times in the state depending on whether both endpoints are in the frontier, one has left, or both have left. I'll also need to track additional information to maintain this invariant. n_closed_endpoints` = number of endpoints that have left the frontier.

Cycle detection: when connecting k to w1 and w2 (|S| = 2):

w1 has label L1, w2 has label L2
If L1 == L2 AND L1 > 0 AND both are in frontier: CYCLE! (they're partners)
If L1 == L2 AND L1 > 0 AND one has left the frontier: still same component, CYCLE! But with my representation, if one has left, the remaining endpoint still has label L1 appearing once. And the closed endpoint had label L1. So if w1.label = 3 and w2.label = 3, and there's only one occurrence of 3 in the state (the other endpoint left), then w1 == w2 (the remaining endpoint). But I said w1 ≠ w2. Contradiction. So if w1 and w2 are both in state, they're different vertices, and if they have the same label, there are 2 occurrences of that label — they're partners and would create a cycle.
If one endpoint of a component left and another endpoint of that same component is still in the frontier, that label appears exactly once. Since w1 and w2 are distinct frontier vertices, they can't both have the same label if it only appears once. So cycle detection is straightforward: connecting k to two frontier vertices with the same positive label creates a cycle if and only if that label appears exactly twice in the current state. with both label = L.

Conclusion: Cycle check = state[w1] = state[w2] = L > 0 and both appear in the frontier. Since by the invariant, each label appears 0, 1, or 2 times, and if L appears once there's only one vertex with that label, if we have w1 ≠ w2 both with label L, then L appears at least twice = exactly twice (by invariant) = w1 and w2 are partners = CYCLE.

So: cycle iff state[w1] == state[w2] and state[w1] > 0. ✓

Now checking the edge cases where one or both vertices have label 0: if either is isolated, they're not in the same component, so no cycle. If both are isolated, they're separate. If they have different positive labels, they're in different components. So the cycle detection rule holds across all cases.

For the state update when connecting k to a subset S, I need to handle the empty set case where k becomes isolated with label 0, and the single-vertex case where k connects to w and they form a new segment with a fresh label. When k connects two isolated vertices, I assign them a fresh label and mark k as interior. If k connects an isolated vertex to an endpoint of an existing component, the isolated vertex becomes the new endpoint while the previous endpoint becomes interior. Now I'm working through the case where k connects two endpoints from different components, which would merge them.

When merging two segments from different components via k, the new merged segment has endpoints at the two non-adjacent endpoints of the original components, while w1, w2, and k all become interior vertices. The tricky part is handling cases where one of those endpoints might have already left the frontier—in that scenario, the component effectively has only one remaining endpoint in the frontier, and I need to account for that when assigning the new label. L1: endpoints w1 (frontier) and e1 (might have left frontier, label = L1)

Component L2: endpoints w2 (frontier) and e2 (might have left frontier, label = L2)
After merging: new component with endpoints e1 and e2
If e1 is in frontier: update state[e1] to new_label
If e1 has left frontier: need to "remember" that this component has a closed endpoint
For the labels: when e1 has left frontier, I track this by having L1 appear only once in state (as state[w1] = L1). After the merge, if I set state[w1] = -1, the label L1 disappears from state. This means I lose track of e1's component membership.

The core problem is that when one endpoint leaves the frontier, I lose the component information when the other endpoint's state changes. So I need a separate tracking mechanism: I'll maintain a counter for each component that records how many endpoints have closed. When merging two components, I combine their closed endpoint counts and assign a fresh label to the merged component. When a frontier vertex leaves, I increment the closed count for its component instead of losing that information.

At the end, I validate that exactly one component exists with the right endpoint configuration—either both endpoints closed, one closed and one frontier, or both frontier—and that all components have merged into a single connected structure.

This is getting intricate but doable. Let me look at the existing C implementation to see if there's something I can adapt rather than building from scratch. The C code uses a frontier-based DP with state map entries and implicit frontier tracking across steps. Reimplementing this in Python would take a couple hundred lines, but it's definitely feasible.

Writing the SMC implementation in Python... Defining the particle state structure...

Now I'm writing a function to generate all valid transitions when a new vertex is introduced, iterating through possible subsets of backward neighbors that can be connected.

Implementing the transition logic... no more forward edges after k is introduced)

So: when we compute the transitions for introducing k:

We process all edges from k to backward neighbors
Then, vertices u with last_fwd[u] == k leave the frontier
Leaving vertex u must have state[u] != 0 (not isolated, since u has no more forward edges)
BUT: vertex k itself might also leave the frontier immediately if last_fwd[k] == 0 (k has no forward edges). In that case:

k is introduced, connected, then immediately leaves the frontier
k must have |S| >= 1 (otherwise k would be isolated when it leaves)
Actually wait: even if last_fwd[k] > 0, k might end up isolated (|S| = 0) temporarily. k will get future connections. So |S| = 0 is valid if k has future edges.

Now I need to think about the edge cases more carefully. If k has no forward edges and no backward neighbors to connect to, then k would be stuck isolated forever—that's invalid. So if last_fwd[k] == 0 and |S| == 0, I need to reject that transition. Similarly, any backward neighbor u that's leaving the frontier (last_fwd[u] == k) but is isolated and not in S would also be invalid, since it would leave the frontier with no connections.

Let me start implementing this logic step by step, handling each constraint as I go.

Writing Python implementation... Defining the PathState class... Continuing with the connection logic...

I'm checking whether vertices in the connection set are valid (not interior vertices), then handling the cycle detection when there are exactly two vertices—if they share the same label, that would create a cycle so I return false. For the actual connection, I'm assigning labels based on the size of the connection set: isolated vertices get label 0, and when connecting to a single vertex, I need to check if that vertex is also isolated. Still writing the relabeling logic... Continuing with the validation logic...

When a vertex closes, if it was an endpoint in an open component, I increment the closed endpoint count for that component. Now I'm checking whether we have a valid terminal state—this requires exactly one component with exactly two endpoints total across all open and closed vertices.

I'm realizing there's a problem with my relabeling approach when merging components. When two endpoints from different components meet, I need to handle the merge more carefully. I need to mark w1 as interior before the relabeling loop to avoid the conflict, and exclude it from iterating over other endpoints that need the new label. I should apply the same fix to w2, though there's another issue I'm starting to see.

When w1 is isolated but w2 is an endpoint of an existing component, the labeling and closed count transfers need to align correctly. I'm also realizing my open_endpoints calculation is flawed—I'm initializing a dict with zeros for each positive label but not actually counting the frontier vertices properly.

Fixing the is_valid_terminal method...

Counting frontier vertices per component label and checking that all components (both open and closed) collapse to exactly one with two total endpoints. I also need to verify there are no remaining isolated vertices in the frontier when the path terminates, since those would indicate an invalid state.

Now I'm setting up the overall SMC algorithm structure...

For each step k, I'm identifying which backward neighbors can be connected, then enumerating all valid connection subsets based on whether k has future edges or must connect to something in the current frontier. Building transition logic...

After applying a transition, I'm closing out any vertices whose last forward edge pointed to the current vertex k, and also closing k itself if it has no outgoing edges. Then I check if the new particle state is valid before adding it to the next generation, multiplying its weight by the branching factor. If all particles die during this step, the algorithm returns zero; otherwise it continues to resampling.

Now I'm checking the effective sample size and doing systematic resampling if it drops below the threshold, then resetting all particle weights equally. For the final estimate, I need to figure out whether to return the sum of weights divided by the number of particles or just the raw sum—the SMC estimate should give me the total count of valid Hamiltonian paths, where each particle tracks a partial path configuration.

The key insight is that at each step k, the expected sum of weights equals the total number of valid partial path configurations on vertices 1 through k, and at the final step n, this becomes the count of complete Hamiltonian paths. So the estimate should be the sum of weights from all particles that reach a valid terminal state.

Now I'm working through the exact weight update mechanics: starting with M particles at weight 1/M each, when a particle has b_k valid transitions at step k, I multiply its weight by b_k and then resample to keep the total weight constant. The final estimate depends on how I normalize—whether I divide by M or use the raw sum of final weights. I'm reconsidering the approach using a product decomposition—breaking down the total count as a product of ratios where each ratio represents the average branching factor at each step. In the SMC implementation, I estimate each ratio by averaging the number of valid transitions across particles at that step, then multiply these estimates together to get the final count. But I need to think through how resampling fits into this calculation properly. the standard SMC weight update.

Standard SMC for counting:

Maintain particles and weights
At each step: w_i^{new} = w_i^{old} × b_k^i (branching factor)
Estimate at step k: N_k = (1/M_0) × sum of w_i^{new} where M_0 = M (initial particle count) Wait, this doesn't account for the start.
Actually let me use the simplest possible version:

Initialize: particles with weight 1.0
At each step k: for each particle, compute b_k^i (valid transitions), sample one, multiply weight by b_k^i.
If b_k^i = 0, particle dies.
At the end: N̂ = sum(weights) × (1 / M) × ... hmm.

Actually, the correct formula: if we start with M particles each with weight 1, and at each step multiply by branching factor, the expected sum of final weights equals N. So the estimate is just the sum of final weights from particles that reach valid terminal states.

Let me check this with a concrete example: for n=2 with a single edge between vertices 1 and 2, there are 2 Hamiltonian paths. At step 1, I introduce vertex 1 with no backward neighbors, so each particle gets weight multiplied by 1. At step 2, vertex 2 has vertex 1 as a backward neighbor, giving one valid transition and weight multiplied by 1 again. Then I need to verify the closing condition for vertices. The final weight formula isn't working out—when M particles all end with weight 1, I get N̂ = M, but that should equal 2, not M. I think the issue is that the DP is counting configurations rather than directed paths, and I need to reconsider what the correct normalization should be.

If the DP counts undirected Hamiltonian paths (N=1 for this graph), then the SMC estimate should also be 1, which means I need to divide the sum of weights by M to get the right answer. Let me verify this against the invariant that E[sum of weights] should equal N × M.

For a simple two-vertex graph, starting with M particles each having weight 1, after one valid transition per particle, the sum stays at M, so N̂ = M/M = 1. Now I'm checking what happens with a more complex graph that has multiple Hamiltonian paths. I'm working through the branching factor calculation more carefully now. When vertex 2 can connect to multiple backward neighbors like vertices 1 and 3, the number of valid connection subsets at that step increases—giving different branching factors at each stage. The key insight is that the algorithm's estimate works because the expected value of the product of branching factors across all particles equals the true count N, which follows from the fundamental SMC identity.

For resampling, I need to track that the total weight stays constant throughout—when I resample M particles proportionally to their weights, each new particle gets assigned weight W/M, preserving the total. This keeps the particles diverse while maintaining the correct weighting.

Now I'm ready to implement the actual algorithm, and I need to handle a few edge cases: the isolation constraint where an empty set S with no forward edges is invalid, vertices with degree zero that can never appear in a Hamiltonian path, and ensuring the path state doesn't create contradictions.

The key insight is that I should validate transitions by checking three conditions: the degree constraint stays within bounds, no cycles form, and critically, any vertex leaving the frontier at this step must have at least one connection to avoid becoming isolated.

For vertex k itself, if it's marked to leave immediately (last_fwd[k] = 0), then S must be non-empty to give k at least one connection—otherwise k would be isolated when it exits the frontier, which is invalid.

Now I'm writing the implementation code... Building the graph structure...

Now I'm defining a particle state class to track connectivity of frontier vertices, where each vertex has a label indicating whether it's isolated, interior to a path, or an open endpoint of some component, along with tracking how many endpoints of each component have already left the frontier.

Initializing the state with empty dictionaries for vertex labels and closed components, plus a counter for generating fresh labels as needed.

Now I'm implementing the core connection logic that handles adding a new vertex and linking it to existing ones, checking for cycles when two vertices share the same component label, and assigning fresh labels to isolated vertices. Still writing the relabeling logic... Checking if the current state represents a valid complete Hamiltonian path by verifying there are no isolated vertices, that all open endpoints belong to a single component, and that exactly two endpoints remain unclosed.

Now I'm defining a function to enumerate valid connection subsets when introducing a new vertex, filtering backward neighbors to exclude interior vertices and building candidate subsets that respect the size and cycle constraints.

Building candidate subsets...

Now I need to filter these candidates based on which nodes are leaving the frontier. If any isolated nodes are about to leave, they must be included in the connection set, otherwise they'd remain disconnected. I'm checking each candidate to ensure all such "leavers" are present, then moving on to the systematic resampling logic.

Implementing systematic resampling... I'm implementing the particle transition logic where each particle samples a valid connection, applies it to create a new state, and then closes any vertices that have reached their final forward position. Writing particle filter logic...

Now I'm spotting a potential bug in how I'm computing the closed targets—I need to make sure I'm only closing backward neighbors of the current vertex k that actually have their forward-last pointer pointing to k, not just any vertex in the label set with that property. me verify the logic once more: when a vertex leaves the frontier, it's either isolated (no connections) or already connected to something in S, and the close_vertex function properly handles both cases by tracking whether it became an endpoint. me trace through again:

enumerate_transitions:

leavers = backward neighbors v of k where last_fwd[v] == k AND state.labels[v] == 0 (currently isolated)
Filter: S must include all leavers (so they get connected before leaving)
After apply_connection:

Leavers that were in S are now connected (label > 0 or interior)
Leavers not in S (well, they're all in S by the filter) — none
Then close_vertex(u) for each u with last_fwd[u] == k:

u might have been originally isolated (0) → now connected (via S) → label > 0 → close OK
u might have been endpoint (> 0) → still endpoint or interior after connection → close OK
u might have been interior (-1) → close OK
Now I'm thinking about the close_targets computation in the main loop. I'm filtering vertices where last_fwd[v] == k, but I need to be careful about whether k itself should be included and whether there could be vertices with last_fwd[u] == k that aren't actually backward neighbors of k. Let me reconsider the closure logic — when I introduce vertex k, I should close all vertices u where their last forward edge points to k, which includes both backward neighbors and potentially k itself depending on its own forward edge state. I need to precompute the backward neighbors of each vertex so I can efficiently close them when their last forward neighbor is introduced. Let me think through the frontier lifecycle: a vertex k enters at step k with a label based on its forward neighbors, then stays in the labels dictionary until step last_fwd[k], at which point all its backward neighbors (and k itself if it has no forward neighbors) get closed out. Now I'm considering the edge case where k is isolated—I need to handle that properly in the closing logic. the empty set case where k has no neighbors at all correctly results in the particle dying. For vertices with only backward neighbors and no forward edges, they must connect to at least one backward neighbor before leaving. I'm also considering a trickier edge case: if multiple isolated vertices all have the same future connection point k, they'd all need to be included in the transition set S, but since |S| ≤ 2, having three or more such vertices would make it impossible to satisfy all connections simultaneously, forcing the particle to die. filter in enumerate_transitions: all(u in S for u in leavers). If |leavers| > 2, no S of size ≤ 2 can contain all of them. So the filter returns empty = particle dies.

Is this correct? If there are 3 isolated leavers of vertex k, can they be connected via k? k can connect to at most 2 of them. The third would remain isolated and leave the frontier as isolated = invalid. So yes, particle must die. ✓

But wait: what if one of the leavers is a backward neighbor of k that hasn't been isolated? I'm only looking at isolated leavers. Non-isolated leavers (endpoints or interior) don't need to be in S — they can leave as endpoints or interior without issues.

OK I think the logic is sound. At the final step k=n, all vertices have been introduced and should have left the frontier by then, since every vertex leaves once its last forward neighbor is introduced. For vertex n itself, there are no forward neighbors, so it leaves immediately at step n. After processing all steps, the labels should be empty, meaning all vertices have been closed out.

Now I'm checking the terminal validity condition: labels must be empty, and the closed components must form exactly one connected component with exactly 2 closed endpoints total.

Validating this with a function that checks for isolated frontier vertices (which would be invalid), counts the open endpoints per component, and ensures all components (open and closed) form a single connected structure with exactly 2 total endpoints.

For a Hamiltonian path on G_n, all vertices close by step n since each vertex v has last_fwd[v] ≤ n, so the labels dictionary empties out and we're left with one closed component containing exactly 2 endpoints—which satisfies the terminal condition. At step n, I introduce the vertex with its backward connections and remove all vertices whose last forward neighbor is n—which empties the frontier. Now I need to think through the branching factors: for each vertex k, its backward neighbors are all u < k where u+k is a perfect square, and I'm checking which squares work for k=57. I'm realizing the backward neighbor count varies across vertices—averaging around 4 given the graph's mean degree—but I'll trust the algorithm to handle this. The key concern is whether the SMC can properly manage isolated vertices that never connect: if a vertex has degree 0, it has no backward neighbors and no forward neighbors, so at that step the algorithm would need candidates to start from, which might not exist.

Let me trace through the edge cases more carefully. If a vertex has only backward neighbors with no forward neighbors, it must connect to at least one of them. But if it has forward neighbors yet all its backward neighbors are already interior to the current frontier, then it enters as an isolated vertex and waits for those forward neighbors to eventually connect it—though I need to verify this doesn't leave disconnected components at the end.

When a vertex closes at its last forward neighbor step, if it's still isolated, the close operation fails and the particle dies, which correctly rejects invalid configurations. Now I'm ready to implement the full algorithm, though I should double-check the logic for identifying which vertices can leave the frontier at each step. The isolated leavers check ensures only disconnected leavers must be in S, while already-connected leavers are handled correctly by close_vertex regardless of their current state. Now I'm ready to write the clean implementation, and I need to decide on the initial particle setup—I'll start all M particles from the same empty state and let them diversify naturally as the frontier expands. I'm also thinking through how to compute the variance of the SMC estimate from the final particle weights.

The issue is that after resampling, all particles have equal weights, so the final variance is zero and doesn't capture the true uncertainty. I'll use a batch means approach instead—running multiple independent SMC chains and computing the standard error from the variance across those chains.

For now, I'll skip the SEM calculation and just report the point estimate, letting users run different seeds to assess variance themselves. I'm planning to write clean, well-documented Python code around 200-250 lines with CLI arguments for the particle count, resampling threshold, random seed, and ground truth value, then test it on the G_57 graph.

Writing the implementation... 8) = O(112).

For M=10,000 and n=57: 10,000 × 57 × 112 ≈ 64M operations. At Python's speed (~10M ops/sec), that's ~6 seconds. That's OK for a first pass.

For M=100,000: ~64 seconds. Still OK.

Let me now write the code. I'll keep it clean and well-documented.

Actually, I realize there's something I missed. The apply_connection for the |S|=2 case has a bug: I relabel the partner of w1, then also try to relabel the partner of w2. But what if w1 and w2 are from the same component (with the same label)? The code should have already caught this as a cycle (L1 == L2 > 0), so this case doesn't arise. ✓

When I do the relabeling loop, I need to be careful not to relabel w1 itself (since I already set self.labels[w1] = -1). After setting that, the loop won't pick up w1 because its label is now -1, not L1. ✓

But if L1 appears in other frontier vertices, the loop will find and relabel them. If L1 doesn't appear anywhere else in the frontier because the other endpoint already left, then the loop finds nothing and I need to update comp_closed accordingly—if L1 was closed, its count is now 1 since the other endpoint departed.

When both endpoints are in the frontier, one becomes -1 and the other gets relabeled to Lnew. ✓

The logic checks out. Now I'm writing the final code, including the copy method for PState and making sure all fields are properly duplicated, especially the dictionaries. I also need to add progress reporting and the CLI.

Wait, I'm noticing a subtle issue with the component count check in is_valid_terminal. When I merge L1 and L2 into Lnew, comp_closed might have entries for both, and my code transfers them correctly to Lnew. But what if there are other components like L3, L4, and so on?

Actually, thinking through this more carefully: at the end of the DP after step n, all vertices are closed, so the frontier should be empty and comp_closed should contain everything. The check for exactly one component is correct—a valid terminal state means the Hamiltonian path exists as a single connected component. ✓

Isolated vertices with label=0 would fail the close_vertex check and get eliminated earlier anyway. ✓

Now I'm ready to write out the complete code, but let me refine the component counting logic one more time. After step n with an empty labels dictionary, I need exactly one component with two endpoints to form a valid Hamiltonian path.

Writing the validation code... each with 2 endpoints means 2 disconnected paths, so that's not a valid Hamiltonian path either. The algorithm checks out. Now I'm ready to implement this, but first I need to think through the computational complexity. For the larger graph with 57 vertices, the DP explores roughly 10^14 valid Hamiltonian paths, so I'll use around 10,000 particles in the SMC to get a reasonable estimate. With an average frontier size of 10-12 and a branching factor of 2-5 at each step, that gives me about 570,000 particle-steps total. The relative error for this setup should scale roughly as the square root of the ratio between steps and particles.

For M=10,000 particles, I'm looking at around 7.5% relative error, which is acceptable for a first pass. I'll start with M=1,000 for quick testing and can scale up if needed.

Now I'm realizing there's a subtle issue in my transition logic. When a vertex is about to leave the frontier and it's currently an endpoint, I need to carefully track what happens if another vertex connects to it. If that connection happens, the endpoint becomes interior and the connecting vertex takes over as the new endpoint for that path. But if no connection happens, the endpoint stays as is and gets registered as closed when it leaves. Let me make sure my enumerate_transitions function handles this correctly.

I'm going to write out the full SMC implementation now. The core structure tracks the frontier vertices with their labels, maintains a count of closed endpoints per component, and iterates through each step where I enumerate valid transitions for each particle, sample one, apply it, and handle vertices leaving the frontier. Dead particles get dropped, and I reweight by the branching factor.

Writing the implementation... I'm setting up preprocessing structures to track forward neighbors, backward neighbors, and when each vertex leaves the frontier during the algorithm's execution.

Now I'm defining a particle state class to track vertex labels, component closure counts, and generate fresh component identifiers as needed.

The copy method duplicates the state by creating new dictionaries for labels and closure counts while preserving the next label counter. The connect method handles adding a vertex to a set of backward neighbors, checking for cycles when connecting to two vertices with the same component label, and assigning appropriate labels based on the cardinality of the neighbor set.

Continuing with the connection logic...

When a component already has a label, I find its other frontier endpoint and relabel it to match the new segment, then transfer any closed-endpoint counts from the old component to the new one. The close method removes a vertex from the frontier and validates whether it was actually part of an active path.

Now I'm checking if the current state represents a valid complete Hamiltonian path by verifying there are no isolated frontier vertices, counting open endpoints per component, and confirming everything belongs to a single component with exactly two endpoints.

I'm writing the transition enumeration function... I'm implementing a resampling function that normalizes particle weights, computes cumulative probabilities, and uses systematic sampling to select indices based on uniform steps. Then I'm starting the main SMC algorithm...

Setting up the sequential Monte Carlo routine to estimate Hamiltonian path counts by initializing particles and weights, precomputing graph structures, and iterating through each frontier step where I resample particles based on backward reachability and vertices leaving the frontier. Writing particle filter logic... Setting up the CLI argument parser with options for the number of particles, resampling threshold, random seed, and the ability to run multiple independent trials to estimate variance. I'm also checking if there's a known ground truth value for the given graph size and then looping through each run to collect estimates.

Now I'm executing the SMC sampler for each trial with an incremented seed, collecting the estimates, and computing summary statistics like mean, standard deviation, and standard error across all runs.

Wait, let me reconsider the frontier exit logic. When I store vertices in leaving_at[k], I'm capturing those with last_fwd[v] = k or vertices with no forward edges that exit at their introduction step.

For any vertex v, if it has no forward edges (last_fwd[v] = 0), it leaves when introduced at step v. If it has forward edges, it leaves when its last forward neighbor is introduced at step k. My precomputation correctly assigns each vertex to the appropriate exit step, so now I need to verify this logic works correctly in the frontier updates. the close step:

python
for u in leaving:  # leaving = leaving_at[k]
    if u in ns.lab:
        if not ns.close(u): ok = False; break
For vertex k itself (if last_fwd[k] = 0, so k ∈ leaving_at[k]):

k should be in ns.lab (we just added it via connect())
close(k) will handle it
For backward vertices v with last_fwd[v] = k:

v should be in ns.lab (it was added when v was introduced)
close(v) will handle it
Looks correct!

But wait: ns.connect(k, S) adds k to ns.lab. But if k ∈ leaving_at[k] (last_fwd[k] = 0), then in the close loop, we try to close k. After ns.connect, k's label depends on S — it's 0 if S is empty, positive if |S|=1, or -1 if |S|=2. Since I filter to require |S| >= 1 when last_fwd[k] = 0, k won't have label 0 at the close step, so closing it should work. The only edge case is if k somehow connects to future vertices despite last_fwd[k] = 0, but that's impossible by definition.

Actually, I realize I need to double-check the iso_leavers logic. When I compute iso_leavers = [v for v in leaving if v != k and lab.get(v, 0) == 0], I'm excluding k itself and only looking at other vertices in leaving. The last_fwd[k] == 0 filter handles k's isolation requirement separately, so this should be fine.

Now I'm checking whether all vertices in leaving_at[k] are actually backward neighbors of k. The set contains k itself (if it's isolated) plus any vertex v < k where (v, k) is an edge, meaning v is a backward neighbor. So the iso_leavers check correctly applies only to those backward neighbors that are isolated.

I think the logic is sound, but I should verify that all vertices in leaving_at[k] are actually in the label map, since a vertex could theoretically be in the leaving set without having been introduced yet.

Actually, by construction, when we process leaving_at[last_fwd[v]], that's when v gets closed. So if v has last_fwd[v] = k, it should still be in the label map at step k since we haven't processed it yet. And even if v has multiple forward edges and we set last_fwd[v] to the maximum, v still gets closed exactly once at that step.

Now I need to handle the connect function for the |S|=2 case, particularly when both endpoints have the same label and belong to the same component.

When w1 and w2 are both endpoints of component L (where L > 0), the cycle check at the beginning catches this immediately since L1 == L2, so the function returns False before processing anything. If both are isolated (L1 = L2 = 0), there's no cycle—I process w1 first (relabeling it to Lnew), then w2 (also relabeling to Lnew), resulting in both having the same new label with k = -1.

Now I'm checking the remaining cases: when w1 is an endpoint of some component L1 > 0 and w2 is isolated (L2 = 0), I relabel w1's partner to Lnew and also assign w2 to Lnew, creating a path segment. The case where both L1 > 0 and L2 > 0 but they're different components is impossible by definition since the same label means the same component.

When both endpoints have positive labels from different components, I process both w1 and w2 by relabeling their respective partners to Lnew and accumulating the component transfers, which merges them into a single segment. But I'm noticing a potential issue: when I search for the partner vertex to relabel, I'm finding the first vertex with the matching label, which could be w1 itself if I haven't already marked it as interior.

The fix is to set the interior label first—marking w1 as -1 before searching for its partner—so the loop skips over w1 and finds p1 instead. My current code does this correctly by setting lab[w] = -1 before the relabeling loop, ensuring the partner gets found and relabeled to Lnew while w1 stays interior.

Now I'm checking whether w2 could be the partner of w1. If w2 were p1, they'd have to share the same component label, but I already verified L1 ≠ L2 at the start, so this can't happen.

One more edge case to consider: what if w is an endpoint whose partner has already closed (meaning cc[L] = 1)? In that scenario, when I search for other vertices with label L in the frontier, I'll only find w itself. After marking w as interior, the loop finds nothing to relabel, which is fine—the closed endpoint is already handled by the cc transfer logic that adds the count to the merged component.

I think the connect function is solid now. Let me write out the final clean implementation.

Wait, I'm realizing a potential issue with the endpoint accounting. When merging two components L1 and L2 into Lnew, I'm adding their endpoint counts together, but each component started with exactly 2 endpoint slots. So Lnew would end up with 4 total slots, which exceeds the 2 slots a single component should have. That doesn't seem right.

Let me think about this differently. When I connect k to w1 and w2, those two endpoints get filled by the new vertex k, leaving only the partners of w1 and w2 as the remaining open endpoints for Lnew. So Lnew should have exactly 2 endpoint slots total—the two remaining open endpoints. p2 (if in frontier) = 0, 1, or 2

Closed endpoints: cc[L1] + cc[L2] (from previously closed endpoints)
When w1 becomes interior, it frees up p1 as L1's remaining endpoint slot, and similarly w2 becoming interior leaves p2 as L2's remaining slot. The merged component Lnew has exactly 2 endpoint slots total from p1 and p2.

Now I need to track how closed endpoints carry over: if p1 is still in the frontier, it gets relabeled as an open endpoint of Lnew, but if p1 already left the frontier (meaning cc[L1] = 1), it's already counted as closed. The same logic applies to p2, so cc[Lnew] = cc[L1] + cc[L2].

For a valid terminal state, open_ep[Lnew] + cc[Lnew] must equal 2, which checks out since we have exactly 2 endpoint slots total. The logic holds up.

I should also verify the edge case where both endpoints of L1 have left the frontier (cc[L1] = 2), meaning L1 doesn't appear in the label list anymore. If that's the case, w1 can't have label L1 in the current labeling, but w1 needs to be in both S and the label list for the merge to work.

Actually, this is impossible—if w1 is in the frontier with label L1, then at least one endpoint of L1 is still in the frontier, so cc[L1] ≤ 1. This contradicts cc[L1] = 2, so the case can't occur.

The algorithm checks out. Now I'm ready to implement the final code.

One issue: the loop that searches for a frontier vertex with a given label is O(|frontier|), but with |frontier| ≤ 14 that's fine. I should test this on a small example first to verify correctness—maybe run the SMC for n=57 and see if the estimate matches the expected ~68 × 10^9 Hamiltonian paths.

Writing the implementation...

Now I'm checking the edge case in the connect function: when I relabel w1's partner, w2 shouldn't accidentally get picked up since they're in different components at that point. The order matters—after processing w1 and relabeling its partner, w2 still has its original label L2 unchanged, so the second relabeling step proceeds correctly.

Writing the final implementation... The next chunk shows example usage for larger problem sizes and documents the command-line options—particle count, number of independent runs for variance estimation, RNG seed, and resampling threshold—along with a table of precomputed exact Hamiltonian path counts for specific graph sizes that can be used for validation.

Then it explains the core algorithm: vertices are introduced sequentially, and the frontier tracks which already-introduced vertices still connect to unintroduced ones. Each particle maintains a partial solution as disjoint path segments, with vertex labels indicating whether they're isolated, interior nodes with degree 2, or endpoints of open segments, and at each step the particle branches over all possible ways to extend these segments.

Now I'm setting up the implementation with the known values for the larger graph sizes and importing the necessary libraries...

Building the graph structure... Defining the particle state class... Writing the connection logic... Still writing the validation logic... Building the candidate subset logic... Implementing systematic resampling... Still writing particle filter logic... Writing the CLI interface... Setting up the estimates list and running the SMC sampler multiple times with different seeds, collecting each result and printing progress updates along the way. Then if there are multiple runs, I'm computing the mean and standard deviation across all estimates.

Now I'm calculating the standard error of the mean and printing a summary with the aggregate statistics, including a z-score comparison to the ground truth if available. After that, I want to test this on a small example to verify it's working correctly.

Looking at the algorithm, I need to check whether the closing logic properly handles vertex k itself—specifically, when k has no forward edges and gets added to the union-find structure, does it correctly close out that vertex in the subsequent loop?

When k is in the leaving set because last_fwd[k] = 0, it gets a label from the connect operation. The close operation then pops that label and either increments the endpoint count (if L > 0), returns false for isolated vertices (which shouldn't happen given our filtering), or handles interior vertices (L = -1). This correctly processes k.

For backward vertices v that point to k as their last forward neighbor, they also leave at step k and get closed after their labels potentially update from the connect operation.

Now I'm tracing through the component relabeling logic more carefully. When connecting a single vertex w that's an endpoint of some component, w becomes interior and k takes over as the new endpoint for that component.

If the other endpoint of that component is still in the frontier, both k and that endpoint now share the same label—two endpoints total. If the other endpoint has already left, then k becomes the sole open endpoint while the closed endpoint count stays at one, still giving us two endpoints for the component overall.

For the two-vertex case where w1 is an endpoint, I'm relabeling w1 as interior and then updating all vertices that share w1's component label to point to the new component label instead.

Handling the closed endpoint case...

When w2 gets processed, its partner p2 could theoretically be w1, but since w1 was already marked as -1 and has a different original label, the relabeling logic correctly avoids conflicts.

After both vertices are processed, the interior vertex k and both frontier vertices w1 and w2 end up with label -1, while their respective partners p1 and p2 get relabeled to Ln with their connected components properly transferred.

Now I'm thinking through the final code implementation and how to handle isolated vertices that remain after processing all steps—specifically, I need to track which vertices are leaving at each step based on their last forward edge, or the vertex itself if it has no forward edges. v gets removed from labels when we process step k = last_fwd[v] and call close(v). After step n, labels should be empty, and the valid_terminal check verifies this by looking for any remaining isolated vertices and checking the connected components. For vertices with no forward edges (last_fwd[v] = 0), they leave at their own introduction step, so they're removed from labels immediately after being connected.

Now I'm ready to write the code. Actually, wait—I need to double-check the iso_lv logic in enum_trans. The check for lab.get(v, 0) == 0 is looking at whether v's current label is 0 (isolated), but v should only be in the leaving set if it's already been introduced. Since leaving_at[k] contains vertices with last_fwd[v] = k where k > v, those vertices have definitely been introduced by that point.

The tricky part is that lab.get(v, 0) returns 0 both when v isn't in lab and when v is in lab with label 0—but v should always be in lab at this point since it was introduced earlier and hasn't been closed yet. I should add a safety check to be sure, but the logic should hold.

Now I'm verifying whether a vertex v in leaving_at[k] could possibly not be in lab. Since v was introduced at step v and stays in lab until step last_fwd[v] = k when it gets closed, v must be in lab before that closing happens. The only edge case is if v has degree 0—then no particles survive to step k anyway, so v wouldn't appear in leaving_at[k] in the first place. And if v has only backward edges, the analysis continues from there.

So the logic checks out: any v in leaving_at[k] with v < k is definitely in lab at step k, and lab.get(v, 0) retrieves the correct label. Now I'm ready to finalize the code and test it on G_4.

Let me trace through the test case with n=4. The edges are formed where u+v equals a perfect square with 1 ≤ u < v ≤ 4. The only perfect square in range is 4, which gives us the edge (1,3). This means vertices 2 and 4 are isolated, so there's no Hamiltonian path possible since we can't connect all four vertices.

Now I'm working through the SMC algorithm step by step. At k=1, vertex 1 has no backward neighbors but its last forward neighbor is 3, so I record that. Since there are no backward neighbors, the only candidate is the empty set. At k=2, vertex 2 also has no backward neighbors, and checking for forward neighbors where 2+v is a perfect square...

actually, that would require v=2 (giving 4), but that's a self-loop which doesn't count. So vertex 2 has no forward neighbors either, meaning all particles die at step 2 and the count is 0. This confirms the algorithm correctly identifies that vertex 2 is isolated. Now I want to trace through G_15 to see a more complex example.

I'll skip the detailed walkthrough and just implement the final code with proper error handling and documentation. I need to verify the weight formula is working correctly—at each step, particles multiply their weights by the branching factor, so after resampling the total weight should still equal M times the number of valid partial paths. When I resample, individual weights get reset but the sum is preserved, which maintains the unbiased estimate.

At the final step, if all terminal particles are valid, then the sum of their weights equals M times the number of Hamiltonian cycles, giving me the right estimate. But I should check whether a particle with non-zero weight could somehow end up in an invalid terminal state—if the DP is correct, all valid partial paths should either complete successfully or get eliminated along the way.

Looking at the frontier structure: once all vertices are introduced and closed, what remains is the connected components map showing which segments exist and how many open endpoints each has. A valid Hamiltonian path requires exactly one component with both endpoints closed. If there are multiple components at the end, the particle represents disconnected path segments rather than a single path, so excluding these from the count is correct. The SMC could generate particles that split into multiple components if the branching factor treats path-extending and path-merging transitions equally, so some invalid terminal states are plausible.

The real concern is efficiency: particles reaching invalid terminal states waste computational effort. The resampling step helps by favoring particles with higher weights, but valid terminal states aren't necessarily weighted higher. For the SMC to be effective, a reasonable fraction of particles need to reach valid terminal states—similar to how the completion rate matters in importance sampling approaches. This becomes especially critical for graphs with higher pathwidth. I'm realizing that particles can die mid-stream when a vertex has no valid backward neighbors to connect to, which seems unlikely for typical vertices in G_n but is theoretically possible. The SMC approach should work reasonably well for this problem, so I'll move forward with testing it. Before diving into the code, I'm wondering if I can streamline the state representation—the current approach with component tracking is getting complex, so I'm considering a union-find structure instead where I track component roots and vertex degrees directly, recording path endpoints as vertices leave the frontier.

Actually, this alternative is essentially equivalent to what I already have, just with different bookkeeping. The dict-based approach is simpler for small frontiers, so I'll stick with my current design and write the final code. One optimization I need to handle: the connect function currently loops through all keys in the label dictionary to find a vertex's partner, which is inefficient.

For small frontiers this is fine, but I should verify the safety of modifying dictionary values during iteration. In Python 3.7+, dictionaries maintain insertion order and allow value modifications during iteration as long as I'm not adding or removing keys. Since I'm only changing values, this is safe. I do set the label to -1 before the loop starts, so I won't accidentally match against the wrong vertex.

The loop finds the partner vertex with the old label and relabels it. The key question is whether that partner could have already been modified in a previous outer loop iteration—but that only happens if the partner is one of the vertices being processed directly, which in the |S|=2 case means it would have to be w1 or w2.

Since we verified that L1 ≠ L2 ensures w1 and w2 are in different components, the partner of w1 can't be w2, so it remains unmodified when we process w1. The logic checks out.

Now I need to verify the valid_terminal check after the final resampling step...

Actually, I'm realizing the weight calculation is more subtle than I initially thought. After the last resampling, weights reset to equal values, but then the final step with branching changes them again. So at the very end, weights aren't necessarily uniform anymore. The estimate should sum the weights of particles that pass the valid_terminal filter, divided by the original particle count M.

Now I'm catching an issue with the resampling logic—I always resample to exactly M particles, but particles can die during transitions, leaving fewer than M alive. If I'm resampling from a reduced population, I need to be careful about how I'm handling that.

Actually, looking at this more carefully: when particles die (no valid transitions), they don't get added to the new population, so the particle count drops below M. The resampling step then duplicates survivors to get back to M particles, which is standard systematic resampling and correct. After resampling, each of the M particles gets weight W/M, preserving the total weight. But I need to verify that my final estimator is computing the sum correctly over only the particles that actually survived to the terminal step. I've confirmed the math is right—particles only contains survivors, dead particles are effectively weighted at zero, and dividing by the original M gives the correct estimate. Now I'm writing out the full implementation with the graph construction, particle filtering, and resampling logic all together.

Writing the complete implementation... I'm building an adjacency list by checking which pairs of numbers sum to perfect squares, then precomputing forward and backward neighbors for each vertex along with tracking when vertices leave the frontier. Now I'm defining a state class that tracks connectivity information for the broken-profile dynamic programming approach, where each frontier vertex has a label indicating whether it's isolated, interior to a path, or an open endpoint.

Implementing the state methods... Continuing with the relabeling logic to merge components, then I'm handling the frontier closure by removing vertices and tra

Show more



Claude is AI and can make mistakes. Please double-check responses.
