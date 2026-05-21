import json
import os
import subprocess
import sys
from pathlib import Path


BASELINES = (
    "sparse_direct",
    "horner_one_step",
    "mv_horner",
    "cse",
    "top_down_search",
)


def test_target_distribution_script_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_jsonl = tmp_path / "targets.jsonl"
    summary_json = tmp_path / "summary.json"
    table_csv = tmp_path / "table.csv"
    latex_output = tmp_path / "table.tex"

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_target_distribution.py",
            "--count",
            "2",
            "--progress",
            "0.0",
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--table-csv",
            str(table_csv),
            "--latex-output",
            str(latex_output),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    for row in rows:
        costs = [row[name] for name in BASELINES]
        assert row["best_baseline"] == min(costs)
        assert set(row["best_baseline_winners"])
        for winner in row["best_baseline_winners"]:
            assert row[winner] == row["best_baseline"]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["total_targets"] == 2
    table = summary["table"]
    assert sum(row["n"] for row in table if row["group_type"] == "progress") == 2
    assert sum(row["n"] for row in table if row["group_type"] == "family") == 2
    assert next(row for row in table if row["group_type"] == "overall")["n"] == 2

    assert table_csv.exists()
    assert "\\begin{tabular}" in latex_output.read_text(encoding="utf-8")
