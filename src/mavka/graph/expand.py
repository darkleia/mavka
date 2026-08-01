from mavka.graph.adjacency import EDGE_ANALOGOUS, EDGE_TEMPORAL

_ALL_EDGE_TYPES = (EDGE_TEMPORAL, EDGE_ANALOGOUS)


def decay_for_depth(depth: int) -> float:
    """Score multiplier for a node found at this many hops from a seed:
    1.0 at depth 0 (a seed itself), 0.5 at depth 1, 0.25 at depth 2, and so
    on (0.5 ** depth) -- a seed counts at full strength, each additional
    hop is progressively weaker evidence. A simple, fixed, documented
    decay, not a learned weighting.
    """
    return 0.5**depth


def _outgoing_edges(graph, node_id: int, edge_types) -> list[tuple[int, int, float]]:
    triples = []
    for edge_type in edge_types:
        for dst, weight in graph.neighbors_of_type(node_id, edge_type):
            triples.append((dst, edge_type, weight))
    return triples


def expand(
    seed_ids: list[int], graph, depth: int, max_nodes: int, edge_types=None
) -> list[tuple[int, dict]]:
    """Breadth-first walk outward from seed_ids over graph's edges, up to
    depth hops, returning every node reached (seeds included) paired with
    its provenance.

    depth = 0 is the off switch: returns exactly the seeds, unchanged.
    depth = 1 adds direct neighbors; depth = 2 adds neighbors-of-neighbors;
    and so on. edge_types restricts which edge kinds to follow (an
    iterable of edge-type codes from mavka.graph.adjacency, e.g. [EDGE_TEMPORAL]);
    None (the default) follows both temporal and analogous edges.

    max_nodes is a hard cap on the total number of nodes returned,
    checked continuously (even mid-level, not just between BFS layers) --
    depth bounds how far the walk can go, max_nodes bounds how wide it can
    get. Seeds are never subject to this cap: all of seed_ids are always
    included, even if that already exceeds max_nodes on its own: the cap
    only ever limits what gets *added* beyond the seeds.

    A node reached by more than one path is only recorded once, at
    whichever depth BFS reaches it first (its shallowest, strongest
    provenance) -- this is also what makes cycles safe: a node already in
    the visited set is never re-queued, so the walk always terminates.

    provenance is a dict: {"is_seed": bool, "depth": int,
    "edge_type": int | None, "weight": float | None}. Seeds have
    depth=0, edge_type=None, weight=None (expand() has no similarity
    score of its own for a seed -- that comes from whatever retrieval
    step produced seed_ids). Non-seed nodes carry the edge_type and
    weight of the specific edge that first discovered them.

    Order (and therefore BFS tie-breaking) is deterministic given the
    same graph and seed_ids: seeds first in the given order, then each
    later ring in the order its edges were stored.
    """
    types_to_follow = _ALL_EDGE_TYPES if edge_types is None else tuple(edge_types)

    provenance: dict[int, dict] = {}
    order: list[int] = []

    for seed in seed_ids:
        if seed not in provenance:
            provenance[seed] = {"is_seed": True, "depth": 0, "edge_type": None, "weight": None}
            order.append(seed)

    frontier = list(order)
    current_depth = 0

    while current_depth < depth and frontier and len(provenance) < max_nodes:
        next_frontier = []
        for node_id in frontier:
            if len(provenance) >= max_nodes:
                break
            for dst, edge_type, weight in _outgoing_edges(graph, node_id, types_to_follow):
                if len(provenance) >= max_nodes:
                    break
                if dst not in provenance:
                    provenance[dst] = {
                        "is_seed": False,
                        "depth": current_depth + 1,
                        "edge_type": edge_type,
                        "weight": weight,
                    }
                    order.append(dst)
                    next_frontier.append(dst)
        frontier = next_frontier
        current_depth += 1

    return [(id_, provenance[id_]) for id_ in order]
