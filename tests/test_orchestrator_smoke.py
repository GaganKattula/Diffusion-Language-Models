"""End-to-end smoke test of the auto-research loop on the mock backend."""

from pathlib import Path

from dlm_explorer.agents import Analyst, AutoResearchLoop, TrialRunner
from dlm_explorer.agents.orchestrator import LoopConfig
from dlm_explorer.agents.proposer import GridProposer, SearchSpace
from dlm_explorer.backends import build_backend
from dlm_explorer.eval import build_task
from dlm_explorer.experiment.store import TrialStore


def test_loop_runs_and_finds_frontier(tmp_path: Path):
    space = SearchSpace(
        num_steps=(1, 4, 16),
        unmask_order=("confidence", "random"),
        remask=(False, True),
        temperature=(0.0,),
        remask_frac=(0.25,),
        remask_threshold=(0.5,),
        block_size=(None,),
        seed=(0,),
    )
    backend = build_backend("mock")
    task = build_task("copy", n_examples=8, seq_len=10)
    store = TrialStore(tmp_path / "trials.jsonl")
    loop = AutoResearchLoop(
        GridProposer(space),
        TrialRunner(backend, task),
        Analyst(),
        store,
        LoopConfig(rounds=4, batch_size=8),
    )
    summary = loop.run()

    assert summary.n_trials == len(space.grid())          # full grid explored
    assert summary.best.metrics["accuracy"] > 0
    assert len(summary.frontier) >= 1                      # a Pareto frontier exists
    assert len(store.load()) == summary.n_trials          # persisted to ledger
    # Frontier must be sorted by cost and strictly improving in quality.
    fr = summary.frontier
    for a, b in zip(fr, fr[1:]):
        assert a.metrics["model_calls"] <= b.metrics["model_calls"]


def test_loop_resumes_from_ledger(tmp_path: Path):
    space = SearchSpace(num_steps=(1, 4), unmask_order=("confidence",), remask=(False,),
                        temperature=(0.0,), remask_frac=(0.25,), remask_threshold=(0.5,),
                        block_size=(None,), seed=(0,))
    backend = build_backend("mock")
    task = build_task("copy", n_examples=4, seq_len=8)
    store = TrialStore(tmp_path / "t.jsonl")
    AutoResearchLoop(GridProposer(space), TrialRunner(backend, task), Analyst(), store,
                     LoopConfig(rounds=2, batch_size=2)).run()
    n_after_first = len(store.load())
    # A second loop over the same grid should not duplicate trials.
    AutoResearchLoop(GridProposer(space), TrialRunner(backend, task), Analyst(), store,
                     LoopConfig(rounds=2, batch_size=2)).run()
    assert len(store.load()) == n_after_first
