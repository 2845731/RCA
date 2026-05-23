# Multi-Agent RCA Prototype

This package implements the proposed OpenRCA research prototype as runnable code.

## Components

- `MetricAgent`: full-day global thresholding, then query-window anomaly segments.
- `TraceAgent`: trace call edges, self-calls, slow/error span evidence, and propagation scores.
- `LogAgent`: lightweight log templating and semantic reason hints.
- `SemanticSampler`: Gleaner-inspired trace/log sampler using EPS diversity and anomaly scores.
- `CausalAgent`: builds an `EvidenceGraph` and ranks root cause candidates.
- `VerifierAgent`: enforces OpenRCA candidate lists and output fields.
- `MemoryStore`: optional frozen/updateable case memory for self-evolving experiments.
- `MetaCausalGraph`: reusable metadata-level KPI-to-reason priors.

## Run

```powershell
python -m rca.run_multi_agent_rca --dataset Bank --start_idx 0 --end_idx 2
```

Outputs are written to `test/multi_agent_rca/{dataset}/` and remain compatible
with `main.evaluate`.

For quick local iteration, disable the large trace/log scans:

```powershell
python -m rca.run_multi_agent_rca --dataset Bank --start_idx 0 --end_idx 2 --no_sampler --no_trace --no_log
```

## Ablations

Use flags such as `--no_trace`, `--no_log`, `--no_sampler`,
`--no_recheck_debate`, `--no_meta`, and `--no_verifier`.

Memory experiments should use train/dev/test splits and only pass
`--update_memory` on the training split. Keep `--use_memory` frozen for test
evaluation to avoid leakage.
