"""The pilot harness: token calibration, the perturbed CPython corpus, questions, the three
arms (RAG, agent with LLM compaction, ledger hybrid), scoring, statistics and the cost planner.

Run it as `python -m context_ledger.eval <command>`. Nothing here makes a network call unless
`--live` is given and the live gate (context_ledger.live.gate) lets it through.
"""
