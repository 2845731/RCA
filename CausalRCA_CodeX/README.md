# CausalRCA_CodeX

`CausalRCA_CodeX` is a modular implementation of the CausalRCA-Flow v3 design in
`../causalrca_flow_v3_complete.md`.

The project keeps the OpenRCA benchmark contract used by
`rca/baseline/rca_agent`:

- inputs are read from `dataset/<dataset>/query.csv` in row order;
- optional labels are read from the matching `record.csv`;
- predictions are emitted in the same JSON-like fields consumed by
  `main.evaluate.evaluate`;
- dataset candidates and prompt-domain knowledge are reused from
  `rca/baseline/rca_agent/prompt/basic_prompt_*.py`.

## Structure

```text
causalrca_codex/
  agents/
    data_agent.py
    association_agent.py
    fault_identification_agent.py
    causal_graph_agent.py
    intervention_agent.py
    counterfactual_agent.py
    evaluation_agent.py
  core/
    dataset.py
    telemetry.py
    graph_ops.py
    reasoning.py
    time_utils.py
  orchestrator.py
  runner.py
  cli.py
```

## Run

From `D:/GitHubDownload/OpenRCA/CausalRCA_CodeX`:

```bash
python run_causalrca_codex.py --dataset Bank --start_idx 0 --end_idx 10
```

Results are written by default to:

```text
D:/GitHubDownload/OpenRCA/test/result/<dataset>/causalrca-codex.csv
```

Per-row diagnostics are written next to the CSV under
`causalrca-codex-diagnostics/`.

## Design Mapping

- `DataAgent`: preprocess, KPI aggregation, global thresholds.
- `AssociationAgent`: Pearl association layer, anomaly/fault segment detection.
- `FaultIdentificationAgent`: cross-layer coarse filtering and reserve pool.
- `CausalGraphAgent`: reversed trace graph plus scheme-F edge weighting.
- `InterventionAgent`: ExplainScore and RootCauseScore.
- `CounterfactualAgent`: ContextualExplainScore and reason identification.
- `EvaluationAgent`: step-wise failure attribution.
- `OrchestratorAgent`: observe, reason, act, evaluate, recover loop with a
  bounded recovery budget.
