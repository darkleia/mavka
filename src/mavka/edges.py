from mavka.graph import EDGE_ANALOGOUS, EDGE_TEMPORAL


class EdgeBuilder:
    """Builds the two kinds of outgoing edges for a record as it is
    inserted:

    - Temporal ("what happened next"): predecessor -> record_id, whenever
      the record has a predecessor in its own episode (via
      log.prev_in_episode). Certain and free -- no search, fixed weight
      temporal_weight, tagged EDGE_TEMPORAL. The first step of an episode
      has no predecessor, so no temporal edge for it -- not an error.
    - Analogous (cross-episode): record_id -> match, for up to
      n_analogous of the most similar records found by searching the
      index on z alone, excluding the record itself and anything from its
      own episode, and only above similarity_threshold. Tagged
      EDGE_ANALOGOUS, weighted by the similarity score. This is a plain
      appearance ("this situation resembles that situation") search, not
      action-conditioned retrieval -- a different concept for a different
      purpose.

    on_insert must be called only after record_id has already been added
    to both log and index (and, if used, after the caller has allocated
    its node in graph via add_node()) -- analogous search only ever sees
    whatever is already in the index at that moment, which is exactly
    what prevents linking a record to one inserted after it.
    """

    def __init__(
        self,
        n_analogous: int = 3,
        similarity_threshold: float = 0.5,
        temporal_weight: float = 1.0,
        fetch_factor: int = 5,
    ):
        self.n_analogous = n_analogous
        self.similarity_threshold = similarity_threshold
        self.temporal_weight = temporal_weight
        self.fetch_factor = fetch_factor

    def on_insert(self, record_id, z, action, episode_id, seq_no, log, index, graph) -> None:
        predecessor = log.prev_in_episode(record_id)
        if predecessor is not None:
            graph.add_edge(
                predecessor.id, record_id, weight=self.temporal_weight, edge_type=EDGE_TEMPORAL
            )

        if index.count <= 1:
            return

        fetch_k = min(index.count, self.n_analogous * self.fetch_factor)
        results = index.search(z, k=fetch_k)

        added = 0
        for match_id, score in results:
            if added >= self.n_analogous:
                break
            if score < self.similarity_threshold:
                break  # results are sorted descending; nothing further qualifies
            if match_id == record_id:
                continue
            if log.get(match_id).episode_id == episode_id:
                continue
            graph.add_edge(record_id, match_id, weight=score, edge_type=EDGE_ANALOGOUS)
            added += 1
