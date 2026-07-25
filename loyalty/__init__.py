"""
loyalty — the shared measurement suite for the secret-loyalty transmission experiment.

Every channel arm (generation->SFT, preferences->DPO, reward model->RAFT, agentic
verifier) trains students and then measures them through THIS package, with identical
prompts, identical judges, and identical statistics. That identity is what makes the
headline deliverable — a ranking of which pipeline stage leaks loyalty most — a real
comparison rather than four unrelated numbers.

Three modules, in the order data flows:

  evals.py     WHAT we measure — eval-set YAML loading, the verbatim AuditBench rubric,
               and repo-relative paths.
  measure.py   HOW we measure — load a policy (+/- LoRA), sample, strip think-tags, judge
               absolutely or paired, over anthropic | openrouter | openai. Three run
               modes: run_policy (GPU), run_pair (API), run_calibrate (API).
  analysis.py  WHAT IT MEANS — JSONL storage, cluster bootstrap / paired deltas /
               within-prompt noise, and the comparison tables.

CLI entry points live in scripts/ — this package contains no argparse.
"""
