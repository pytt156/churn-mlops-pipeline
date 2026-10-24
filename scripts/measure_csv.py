"""Compare CSV validation at two commits without changing the working tree."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
AFTER = "5a38e70"
BEFORE = f"{AFTER}^"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def measure(revision, directory):
    commit = git("rev-parse", revision).decode().strip()
    for name in ("kontroller/validera.py", "kontroller/schema.py", "data/kundbas.csv"):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", f"{commit}:{name}"))

    data = directory / "data"
    (data / "empty.csv").write_bytes(b"")
    header = (data / "kundbas.csv").read_bytes().splitlines(keepends=True)[0]
    (data / "headers.csv").write_bytes(header)
    (data / "malformed.csv").write_text('kund_id,alder\n100001,"75\n', encoding="utf-8")
    cases = [
        ("valid", "kundbas.csv", 0),
        ("missing", "missing.csv", 1),
        ("empty", "empty.csv", 1),
        ("headers", "headers.csv", 1),
        ("malformed", "malformed.csv", 1),
    ]
    results = []
    for case, filename, expected in cases:
        run = subprocess.run(
            [sys.executable, "-B", "-m", "kontroller.validera", f"data/{filename}"],
            cwd=directory, capture_output=True, text=True, check=False,
        )
        results.append({
            "case": case, "expected_exit": expected, "exit_code": run.returncode,
            "traceback": "Traceback (most recent call last)" in run.stdout + run.stderr,
            "stdout": run.stdout, "stderr": run.stderr,
        })
    return {"commit": commit, "results": results}


def main():
    with TemporaryDirectory(prefix="csv-measure-") as temporary:
        report = {
            label: measure(revision, Path(temporary) / label)
            for label, revision in (("before", BEFORE), ("after", AFTER))
        }
    lines = [
        "## CSV delivery comparison", "",
        "Retrospective comparison of both versions in the same Python environment.",
        "Only the versioned validator, schema and customer CSV are needed for these cases.",
        "This measures validation behavior, not pipeline runtime.", "",
    ]
    for label, version in report.items():
        lines.append(f"{label}: `{version['commit']}`")
    lines += ["", "| Case | Before exit | After exit | Before traceback | After traceback |",
              "|---|---:|---:|---|---|"]
    for before, after in zip(report["before"]["results"], report["after"]["results"]):
        lines.append(
            f"| {before['case']} | {before['exit_code']} | {after['exit_code']} | "
            f"{before['traceback']} | {after['traceback']} |"
        )
    summary = "\n".join(lines) + "\n"
    print(summary)
    output = ROOT / "measurement-results"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "summary.md").write_text(summary, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as file:
            file.write(summary)
    # Unexpected behavior is preserved in the report and also fails the measurement job.
    return int(any(
        result["exit_code"] != result["expected_exit"]
        for version in report.values() for result in version["results"]
    ))


if __name__ == "__main__":
    sys.exit(main())
