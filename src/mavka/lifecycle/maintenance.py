from mavka.lifecycle.feedback import drain_feedback


class MaintenanceWorker:
    """Single owner of the background lifecycle flow -- the piece that
    was missing before this module existed. compact(), EvictionPolicy,
    FeedbackBuffer, and TieredStore were each unit-tested in isolation
    but never actually driven against a live Memory: nobody drained
    feedback into an eviction policy, nobody triggered compaction or
    migration during a real run. This worker is that missing owner. It
    runs entirely off the hot path -- call step() explicitly (from an
    eval harness or a loop), never scheduled or triggered internally.

    step() performs, in this fixed order, only the pieces that are
    configured (each independently gated -- a worker can be built to run
    just one or two of these):
    1. Drain feedback -- drain_feedback(feedback_buffer, eviction_policy):
       apply buffered "this memory helped" events to the eviction
       policy's utility signal. Requires both feedback_buffer and
       eviction_policy.
    2. Compaction -- memory.compact(): merge near-duplicates / drop
       tombstones. Runs once memory.count has reached
       compaction_threshold (a simple "don't bother on a nearly-empty
       memory yet" cadence gate, not a tombstone/duplicate counter).
       Requires compaction_threshold to be set.
    3. Eviction -- memory.evict(capacity): tombstone the lowest-keep-score
       non-pinned records down to capacity, if over it (evict_to_capacity
       is itself a no-op under capacity). Requires both eviction_policy
       and capacity.
    4. Migration -- memory.migrate(): move aged hot records to cold.
       Runs whenever memory.use_tiering is True.

    Composability, discovered while wiring this (see Memory.compact()'s
    docstring for the full detail, and this task's summary for the
    writeup): compaction and eviction both rebuild the log and reassign
    ids, which (a) only composes with a z-only (action_scale=0.0) index
    -- compact()'s rebuild always re-inserts plain z vectors, so an
    action-conditioned index raises immediately -- and (b) is
    structurally incompatible with tiering, since TieredStore's ids must
    never be reassigned. Memory.compact()/.evict() refuse both cases
    with a clear error rather than silently producing a broken index or
    corrupting TieredStore's tier bookkeeping. Practically: a single
    Memory typically uses either compaction+eviction (plain index) or
    migration (tiering), not both -- this worker still supports either
    combination, it is the caller's configuration that decides which
    pieces actually fire.
    """

    def __init__(
        self,
        memory,
        *,
        eviction_policy=None,
        feedback_buffer=None,
        capacity=None,
        compaction_threshold=None,
        enabled=True,
    ):
        self.memory = memory
        self.eviction_policy = eviction_policy
        self.feedback_buffer = feedback_buffer
        self.capacity = capacity
        self.compaction_threshold = compaction_threshold
        self.enabled = enabled

    def step(self, now_ns=None) -> dict:
        """One maintenance pass. Returns a report of what it did:
        {"feedback_events_applied": int, "compaction": dict | None,
        "eviction": dict | None, "migration": dict | None} -- a None
        value means that stage didn't run this pass (not configured, or
        nothing to do); feedback_events_applied is 0 the same way.
        A no-op, all-zero/None report if disabled.
        """
        report = {
            "feedback_events_applied": 0,
            "compaction": None,
            "eviction": None,
            "migration": None,
        }
        if not self.enabled:
            return report

        if self.feedback_buffer is not None and self.eviction_policy is not None:
            report["feedback_events_applied"] = drain_feedback(
                self.feedback_buffer, self.eviction_policy, now_ns=now_ns
            )

        if self.compaction_threshold is not None and self.memory.count >= self.compaction_threshold:
            result = self.memory.compact()
            report["compaction"] = dict(result["stats"])

        if self.eviction_policy is not None and self.capacity is not None:
            result = self.memory.evict(self.capacity, now_ns=now_ns)
            report["eviction"] = {"evicted_ids": result["evicted_ids"], **result["stats"]}

        if self.memory.use_tiering:
            result = self.memory.migrate(now_ns=now_ns)
            report["migration"] = result

        return report
