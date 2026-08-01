import numpy as np

_EMPTY = -1

EDGE_TEMPORAL = 0
EDGE_ANALOGOUS = 1


class AdjacencyStore:
    """Fixed-degree outgoing-edge storage: for each record id, up to
    `degree` slots, each holding the id of a linked record plus a weight.

    This is storage only -- nothing here decides which edges to create or
    how to traverse them. Node ids are meant to line up 1:1 with the log's
    record ids (the same id-alignment discipline as elsewhere in Mavka);
    the caller is responsible for calling add_node() in the same order the
    log assigns ids.

    Backed by three contiguous [capacity, degree] arrays, preallocated and
    grown by doubling (the same pattern as FlatIndex): _neighbors
    (int64, sentinel -1 means "empty slot"), _weights (float32, one weight
    per slot, 0.0 for empty slots), and _edge_types (int8, an arbitrary
    caller-defined code per slot, e.g. EDGE_TEMPORAL/EDGE_ANALOGOUS,
    0 for empty slots -- not encoded into the weight's sign or range,
    since similarity-score weights can themselves be negative).

    Policies:
    - No duplicate edges: adding an edge to a dst_id already present in
      src_id's slots updates that slot's weight instead of using a new
      slot.
    - No self-loops: add_edge(src, src, ...) is always rejected (returns
      False), by default -- fixed-degree storage with no obvious use for
      a node linking to itself.
    - Fixed degree: once all `degree` slots are full, a new edge only
      replaces the current weakest (lowest-weight) slot, and only if the
      new edge's weight is strictly higher than that slot's weight;
      otherwise the new edge is rejected. Ties do not replace.
    """

    def __init__(self, degree: int, initial_capacity: int = 1024):
        self.degree = degree
        self._capacity = initial_capacity
        self._count = 0
        self._neighbors = np.full((initial_capacity, degree), _EMPTY, dtype=np.int64)
        self._weights = np.zeros((initial_capacity, degree), dtype=np.float32)
        self._edge_types = np.zeros((initial_capacity, degree), dtype=np.int8)

    @property
    def count(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count

    def _validate_id(self, id_: int) -> None:
        if id_ < 0 or id_ >= self._count:
            raise ValueError(f"id {id_!r} out of range [0, {self._count})")

    def add_node(self) -> int:
        self._ensure_capacity(self._count + 1)
        id_ = self._count
        self._count += 1
        return id_

    def add_edge(self, src_id: int, dst_id: int, weight: float = 1.0, edge_type: int = 0) -> bool:
        self._validate_id(src_id)
        self._validate_id(dst_id)

        if src_id == dst_id:
            return False

        row = self._neighbors[src_id]
        weights_row = self._weights[src_id]
        types_row = self._edge_types[src_id]

        existing = np.where(row == dst_id)[0]
        if existing.size > 0:
            weights_row[existing[0]] = weight
            types_row[existing[0]] = edge_type
            return True

        empty = np.where(row == _EMPTY)[0]
        if empty.size > 0:
            slot = empty[0]
            row[slot] = dst_id
            weights_row[slot] = weight
            types_row[slot] = edge_type
            return True

        weakest_slot = int(np.argmin(weights_row))
        if weight > weights_row[weakest_slot]:
            row[weakest_slot] = dst_id
            weights_row[weakest_slot] = weight
            types_row[weakest_slot] = edge_type
            return True

        return False

    def neighbors(self, src_id: int) -> list[int]:
        self._validate_id(src_id)
        row = self._neighbors[src_id]
        return [int(x) for x in row if x != _EMPTY]

    def neighbor_weights(self, src_id: int) -> list[tuple[int, float]]:
        self._validate_id(src_id)
        row = self._neighbors[src_id]
        weights_row = self._weights[src_id]
        return [
            (int(row[i]), float(weights_row[i])) for i in range(self.degree) if row[i] != _EMPTY
        ]

    def neighbors_of_type(self, src_id: int, edge_type: int) -> list[tuple[int, float]]:
        self._validate_id(src_id)
        row = self._neighbors[src_id]
        weights_row = self._weights[src_id]
        types_row = self._edge_types[src_id]
        return [
            (int(row[i]), float(weights_row[i]))
            for i in range(self.degree)
            if row[i] != _EMPTY and types_row[i] == edge_type
        ]

    def has_edge(self, src_id: int, dst_id: int) -> bool:
        self._validate_id(src_id)
        self._validate_id(dst_id)
        return bool(np.any(self._neighbors[src_id] == dst_id))

    def degree_of(self, src_id: int) -> int:
        self._validate_id(src_id)
        return int(np.sum(self._neighbors[src_id] != _EMPTY))

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = max(self._capacity, 1)
        while new_capacity < required:
            new_capacity *= 2

        new_neighbors = np.full((new_capacity, self.degree), _EMPTY, dtype=np.int64)
        new_neighbors[: self._count] = self._neighbors[: self._count]
        new_weights = np.zeros((new_capacity, self.degree), dtype=np.float32)
        new_weights[: self._count] = self._weights[: self._count]
        new_edge_types = np.zeros((new_capacity, self.degree), dtype=np.int8)
        new_edge_types[: self._count] = self._edge_types[: self._count]

        self._neighbors = new_neighbors
        self._weights = new_weights
        self._edge_types = new_edge_types
        self._capacity = new_capacity
