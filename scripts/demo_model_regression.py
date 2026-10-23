"""Reproducerbar före/efter-demonstration av regressionsgrinden.

Kör i projektroten: python scripts/demo_model_regression.py
Skapar endast en rapport under docs/; testartefakterna ligger temporärt.
"""

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

from churn.data import DATAKATALOG
from churn.evaluate import main as old_gate
from churn.model import DATAFIL, bygg_modell, hamta_tranings_och_testdata
from churn.regression_gate import kontrollera_regression

ROOT = Path.cwd()
REFERENCE = ROOT / "reference"
REPORT = ROOT / "docs" / "model-regression-before-after.md"


def main() -> None:
    if not (REFERENCE / "modell.joblib").exists():
        raise FileNotFoundError("Saknar reference/modell.joblib")
    reference_meta = json.loads((REFERENCE / "matvarden.json").read_text(encoding="utf-8"))
    current_hash = hashlib.sha256((DATAKATALOG / DATAFIL).read_bytes()).hexdigest()
    if reference_meta["data_sha256"] != current_hash:
        raise ValueError("Referensens datahash matchar inte aktuell data")

    X_train, X_test, y_train, y_test = hamta_tranings_och_testdata(DATAFIL)
    # Avsiktligt svagare kandidat för demonstration, inte för produktionsbruk.
    challenger = bygg_modell()
    challenger.set_params(klassificerare__C=1e-6)
    challenger.fit(X_train, y_train)
    challenger_auc = float(roc_auc_score(y_test, challenger.predict_proba(X_test)[:, 1]))
    challenger_accuracy = float(accuracy_score(y_test, challenger.predict(X_test)))

    with tempfile.TemporaryDirectory(prefix="churn-regression-demo-") as temp:
        output_dir = Path(temp) / "outputs"
        output_dir.mkdir()
        joblib.dump(challenger, output_dir / "modell.joblib")
        candidate_meta = {
            "roc_auc": challenger_auc,
            "accuracy": challenger_accuracy,
            "commit": "demo-only-not-for-release",
            "datafil": DATAFIL,
            "data_sha256": current_hash,
        }
        (output_dir / "matvarden.json").write_text(
            json.dumps(candidate_meta, indent=2), encoding="utf-8"
        )
        # Före: samma kod som befintlig absolutgrind, med standardtröskeln 0.70.
        # main läser MIN_ROC_AUC från miljön; fastställ att den inte är åsidosatt.
        import os

        original = os.environ.pop("MIN_ROC_AUC", None)
        try:
            before_exit = old_gate([str(output_dir / "matvarden.json")])
        finally:
            if original is not None:
                os.environ["MIN_ROC_AUC"] = original

        # Efter: kör den riktiga jämförelsen mot versionshanterad Champion.
        after_pass = kontrollera_regression(REFERENCE, output_dir)

    champion_auc = float(reference_meta["roc_auc"])
    print(f"Före: {'GODKÄND' if before_exit == 0 else 'STOPPAD'} (exit {before_exit})")
    print(f"Efter: {'GODKÄND' if after_pass else 'STOPPAD'}")
    print(f"Champion={champion_auc:.6f}, Challenger={challenger_auc:.6f}")

    if not (before_exit == 0 and not after_pass):
        raise AssertionError("Demonstrationen gav inte väntad skillnad; ingen rapport sparades")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = f"""\
# Före och efter: Model regression gate

Körning (UTC): {datetime.now(UTC).isoformat(timespec="seconds")}

## Problem

Den tidigare kvalitetsgrinden kontrollerade bara ROC AUC >= 0,70.
En modell kunde därmed vara sämre än den godkända modellen och ändå passera.

## Metod

Testkandidaten tränades på samma data och samma train/test-uppdelning
som Champion, med `LogisticRegression(C=1e-6)` i samma förbehandlingspipeline.

Detta är en avsiktligt svagare demonstrationsmodell, inte en föreslagen release.
Kandidatens modellfil och metadata skapades i en temporär katalog, utan att
ändra `outputs/` eller `reference/`.

Båda kvalitetsgrindarna kördes på kandidatens verkligt beräknade mätvärden.

## Resultat

| Kontroll | Resultat |
|---|---|
| Champion ROC AUC | {champion_auc:.6f} |
| Challenger ROC AUC | {challenger_auc:.6f} |
| Skillnad | {challenger_auc - champion_auc:+.6f} |
| Före: absolutgrind >= 0,70 | {"Godkänd" if before_exit == 0 else "Stoppad"} (exit {before_exit}) |
| Efter: Champion–Challenger | {"Godkänd" if after_pass else "Stoppad"} |

Referenscommit: `{reference_meta["commit"]}`
Data SHA256: `{current_hash}`

## Slutsats och begränsning

Den nya grinden stoppar denna avsiktligt försämrade kandidat, som den gamla
grinden släppte igenom.

Detta demonstrerar kontrolllogik, inte en statistisk garanti för förbättrad
generalisering. Testuppsättningen används också av träningsflödet, vilket
begränsar användbarheten som oberoende slututvärdering.
"""

    REPORT.write_text(report, encoding="utf-8")

    print(f"Rapport sparad: {REPORT}")


if __name__ == "__main__":
    main()
