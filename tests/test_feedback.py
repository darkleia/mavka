import threading

import numpy as np
import pytest

from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import split_episodes
from mavka.lifecycle.eviction import EvictionPolicy
from mavka.lifecycle.feedback import FeedbackBuffer, drain_feedback
from mavka.storage.log import AppendLog
from mavka.pipeline import ActionConditionedPipeline
from mavka.index.flat import FlatIndex
from mavka.retrieval.trigger import SurpriseTrigger, evaluate_gated

NOW_NS = 1_000_000_000_000
SECOND_NS = 10**9


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _rescue_scenario():
    """Fresh log/index each call -- evict_to_capacity mutates its log via
    tombstone(), so re-running a second eviction against the same log
    instance would double-apply.
    """
    dim = 8
    log = AppendLog(dim=dim)
    old_ts = NOW_NS - 1000 * SECOND_NS

    # Lowest id -- evict_to_capacity's tie-break drops the lowest id first
    # among equal keep_scores, so this is exactly the one a no-feedback
    # baseline evicts.
    id_rescued = log.append(z=_rand(dim, 0), episode_id=0, pred_err=0.1, timestamp_ns=old_ts)
    other_ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=0.1, timestamp_ns=old_ts)
        for i in range(1, 8)
    ]
    all_ids = [id_rescued, *other_ids]

    index = FlatIndex(dim=dim)
    for id_ in all_ids:
        index.add(log.get(id_).z)

    return log, index, id_rescued, all_ids


def test_loop_updates_utility():
    dim = 8
    log = AppendLog(dim=dim)
    id_helpful = log.append(z=_rand(dim, 0), episode_id=0, pred_err=0.1, timestamp_ns=NOW_NS)
    id_never = log.append(z=_rand(dim, 1), episode_id=0, pred_err=0.1, timestamp_ns=NOW_NS)

    buffer = FeedbackBuffer()
    policy = EvictionPolicy()

    for _ in range(5):
        token = buffer.record_used([id_helpful])
        buffer.tag_outcome(token, helped=True)
    for _ in range(5):
        token = buffer.record_used([id_never])
        buffer.tag_outcome(token, helped=False)

    n_applied = drain_feedback(buffer, policy, now_ns=NOW_NS)
    assert n_applied == 10

    scores = policy.compute_keep_scores([log.get(id_helpful), log.get(id_never)], now_ns=NOW_NS)
    assert scores[id_helpful] > scores[id_never]


def test_utility_rescues_on_eviction_payoff():
    dim = 8

    log, index, id_rescued, all_ids = _rescue_scenario()
    capacity = len(all_ids) - 1

    baseline_policy = EvictionPolicy()
    baseline_result = baseline_policy.evict_to_capacity(
        log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=capacity, now_ns=NOW_NS
    )
    # Sanity check: without feedback, the rescued record WOULD be evicted --
    # otherwise the "survives with feedback" assertion below would be
    # meaningless (could pass by luck of the tie-break rather than because
    # the feedback loop did anything).
    assert id_rescued in baseline_result["evicted_ids"]

    log2, index2, id_rescued2, all_ids2 = _rescue_scenario()
    buffer = FeedbackBuffer()
    fed_policy = EvictionPolicy()

    # Rarely retrieved (once) but consistently helpful when it was.
    token = buffer.record_used([id_rescued2])
    buffer.tag_outcome(token, helped=True)
    n_applied = drain_feedback(buffer, fed_policy, now_ns=NOW_NS - 10 * SECOND_NS)
    assert n_applied == 1

    fed_result = fed_policy.evict_to_capacity(
        log2, index2, index_factory=lambda: FlatIndex(dim=dim), capacity=capacity, now_ns=NOW_NS
    )
    assert id_rescued2 not in fed_result["evicted_ids"]


def test_hot_path_never_calls_policy_directly():
    calls = []

    class SpyPolicy:
        def record_retrieval_feedback(self, id, helped, now_ns=None):
            calls.append((id, helped))

    buffer = FeedbackBuffer()
    spy = SpyPolicy()

    token = buffer.record_used([1, 2, 3])
    buffer.tag_outcome(token, helped=True)
    # Buffering alone -- the hot-path operations -- must never reach the
    # policy.
    assert calls == []

    n_applied = drain_feedback(buffer, spy)
    assert calls == [(1, True), (2, True), (3, True)]
    assert n_applied == 3


def test_batched_matches_immediate_application():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=float(i) * 0.05, timestamp_ns=NOW_NS)
        for i in range(6)
    ]

    events = [(ids[0], True), (ids[1], False), (ids[2], True), (ids[0], True), (ids[3], False)]

    immediate_policy = EvictionPolicy()
    for id_, helped in events:
        immediate_policy.record_retrieval_feedback(id_, helped=helped, now_ns=NOW_NS)

    buffer = FeedbackBuffer()
    batched_policy = EvictionPolicy()
    for id_, helped in events:
        token = buffer.record_used([id_])
        buffer.tag_outcome(token, helped=helped)
    drain_feedback(buffer, batched_policy, now_ns=NOW_NS)

    records = [log.get(id_) for id_ in ids]
    immediate_scores = immediate_policy.compute_keep_scores(records, now_ns=NOW_NS)
    batched_scores = batched_policy.compute_keep_scores(records, now_ns=NOW_NS)
    assert immediate_scores == pytest.approx(batched_scores)


def test_drain_empties_and_is_idempotent():
    buffer = FeedbackBuffer()
    assert buffer.drain() == []

    token = buffer.record_used([1, 2])
    buffer.tag_outcome(token, helped=True)

    first = buffer.drain()
    assert sorted(first) == [(1, True), (2, True)]

    second = buffer.drain()
    assert second == []


def test_gated_off_steps_record_nothing():
    dim = 16
    action_dim = 4
    n_episodes = 10
    episode_length = 15
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=70)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=70)

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=70)
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim + action_dim)
    )
    # High lam -> most steps are gated off (never retrieve).
    trigger = SurpriseTrigger(smoothing=0.1, lam=1000.0, warmup=0)
    buffer = FeedbackBuffer()

    result = evaluate_gated(
        memory_episodes,
        eval_episodes,
        pipeline,
        adapter,
        trigger,
        k=k,
        scale=1.0,
        feedback_buffer=buffer,
    )

    events = buffer.drain()
    retrieved_steps = round(result["retrieval_rate"] * result["n_steps"])
    assert retrieved_steps < result["n_steps"]
    # Every retrieved step contributes exactly k ids (memory is large
    # enough to always return k); gated-off steps contribute none at all.
    assert len(events) == retrieved_steps * k


def test_thread_safety_concurrent_record_and_drain():
    buffer = FeedbackBuffer()
    n_threads = 8
    n_per_thread = 200
    collected = []
    collected_lock = threading.Lock()
    stop = threading.Event()

    def producer(offset):
        for i in range(n_per_thread):
            token = buffer.record_used([offset * n_per_thread + i])
            buffer.tag_outcome(token, helped=True)

    def drainer():
        while not stop.is_set():
            events = buffer.drain()
            if events:
                with collected_lock:
                    collected.extend(events)

    drain_thread = threading.Thread(target=drainer)
    drain_thread.start()

    producers = [threading.Thread(target=producer, args=(i,)) for i in range(n_threads)]
    for t in producers:
        t.start()
    for t in producers:
        t.join()

    stop.set()
    drain_thread.join()
    collected.extend(buffer.drain())

    expected_total = n_threads * n_per_thread
    assert len(collected) == expected_total
    assert len({id_ for id_, _ in collected}) == expected_total


def test_determinism():
    def run():
        buffer = FeedbackBuffer()
        policy = EvictionPolicy()
        for i in range(20):
            token = buffer.record_used([i % 5])
            buffer.tag_outcome(token, helped=(i % 3 == 0))
        drain_feedback(buffer, policy, now_ns=NOW_NS)

        dim = 8
        log = AppendLog(dim=dim)
        ids = [
            log.append(z=_rand(dim, i), episode_id=0, pred_err=0.1, timestamp_ns=NOW_NS)
            for i in range(5)
        ]
        return policy.compute_keep_scores([log.get(id_) for id_ in ids], now_ns=NOW_NS)

    assert run() == run()
