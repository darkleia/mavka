import itertools
import threading


class FeedbackBuffer:
    """Connects retrieval outcomes back to EvictionPolicy.utility, as a
    three-stage deferred loop that keeps the hot path cheap:

    1. record_used(ids) -- the hot-path hook. Call this right after
       recall/recall_scored returns. A plain append under a lock, nothing
       else: no scoring, no searching, no policy calls. Returns a token
       identifying this retrieval event, to pass to tag_outcome() once
       the step's outcome is known.
    2. tag_outcome(token, helped) -- call once "did this retrieval help?"
       can be judged (e.g. after comparing memory-augmented vs
       model-alone prediction error for that step). Moves the event from
       "pending" to "ready to drain" with its verdict attached. Still no
       policy call here.
    3. drain() -- the background-worker hook. Empties and returns
       whatever (id, helped) pairs are ready. This module never calls
       EvictionPolicy.record_retrieval_feedback itself; see
       drain_feedback() for the function that actually applies drained
       events -- that is the only place that call happens.

    All ids used() on a single record_used() call currently share the one
    verdict tagged for that step (per-id credit assignment -- e.g.
    weighting which of several retrieved ids actually mattered for the
    outcome -- is a deliberate later refinement; tag_outcome's single
    token->verdict mapping is the hook it would plug into).

    Thread-safe: record_used, tag_outcome, and drain all take the same
    lock, so a hot-path caller and a background drainer can run
    concurrently without corrupting the buffer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._next_token = itertools.count()
        self._pending: dict[int, list[int]] = {}
        self._ready: list[tuple[int, bool]] = []

    def record_used(self, ids: list[int]) -> int:
        token = next(self._next_token)
        with self._lock:
            self._pending[token] = list(ids)
        return token

    def tag_outcome(self, token: int, helped: bool) -> None:
        with self._lock:
            ids = self._pending.pop(token, None)
            if ids is None:
                return
            self._ready.extend((id_, helped) for id_ in ids)

    def drain(self) -> list[tuple[int, bool]]:
        with self._lock:
            ready, self._ready = self._ready, []
        return ready


def drain_feedback(buffer: FeedbackBuffer, policy, now_ns: int | None = None) -> int:
    """Background-worker hook: pull whatever (id, helped) events are
    ready in buffer and apply each to policy via
    policy.record_retrieval_feedback -- the only place that method is
    ever called (never from a retrieval/hot path). Returns how many
    events were applied. Safe on an empty buffer (returns 0); since
    drain() empties the buffer, calling this again immediately with no
    new events in between is a no-op.
    """
    events = buffer.drain()
    for id_, helped in events:
        policy.record_retrieval_feedback(id_, helped=helped, now_ns=now_ns)
    return len(events)
