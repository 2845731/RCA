"""Debug script to investigate Case 4 ranking issue."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "CausalRCA_CodeX")

from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.core.dataset import build_query, dataset_path
from causalrca_codex.orchestrator import OrchestratorAgent
import pandas as pd

config = AgentLoopConfig()
ds_dir = dataset_path(config, "Bank")
query_df = pd.read_csv(ds_dir / "query.csv")

# Case 4 (Row 4): Tomcat04 is ground truth
row = query_df.iloc[4]
query = build_query(
    config=config, dataset="Bank", row_id=4,
    task_index=str(row["task_index"]),
    instruction=str(row["instruction"]),
    scoring_points=str(row["scoring_points"]),
)

orch = OrchestratorAgent(config)
result = orch.run(query)

# Print key outputs
ws = orch.workspace
scores = ws["association_layer"].get("anomaly_scores", {})
first_ts = ws["association_layer"].get("first_anomaly_ts", {})
ranking = ws["intervention_layer"].get("ranking", [])

print("\n=== Anomaly Scores (top 10) ===")
for comp, score in sorted(scores.items(), key=lambda x: -x[1])[:10]:
    ts = first_ts.get(comp, "N/A")
    print(f"  {comp}: score={score:.4f} first_ts={ts}")

print(f"\n=== InterventionAgent Ranking (top 10) ===")
for i, row in enumerate(ranking[:10]):
    print(f"  #{i+1} {row['component']}: ES={row['ExplainScore']:.4f} RCS={row['RootCauseScore']:.4f}")

print(f"\n=== Ground Truth: Tomcat04 ===")
print(f"  Tomcat04 anomaly score: {scores.get('Tomcat04', 'NOT FOUND')}")
print(f"  Tomcat04 first_ts: {first_ts.get('Tomcat04', 'N/A')}")
