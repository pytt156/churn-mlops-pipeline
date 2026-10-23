"""Champion–Challenger-grind före leverans.

Kör från projektroten: python -m churn.regression_gate

Jämför två betrodda modellartefakter på samma testobservationer. Den befintliga
absoluta ROC AUC-gränsen kontrolleras separat av churn.evaluate.
"""

import hashlib
import json
import logging
import math
import sys
import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import roc_auc_score

from churn.data import DATAKATALOG
from churn.model import hamta_tranings_och_testdata

REFERENCE = Path("reference")
OUTPUTS = Path("outputs")
METRIC_TOLERANCE = 1e-10  # Endast flyttalsavrundning, inte tillåten modellregression.

logger = logging.getLogger(__name__)


def godkann_modell(champion_auc: float, challenger_auc: float) -> bool:
    """Godkänn endast ändliga ROC AUC-värden utan försämring."""
    return (
        math.isfinite(champion_auc)
        and math.isfinite(challenger_auc)
        and challenger_auc >= champion_auc
    )


def _las_metadata(sokvag: Path) -> dict:
    metadata = json.loads(sokvag.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata måste vara ett JSON-objekt: {sokvag}")
    return metadata


def _kontrollera_sparat_auc(namn: str, beraknat_auc: float, sparat_varde: object) -> None:

    if isinstance(sparat_varde, bool) or not isinstance(sparat_varde, (int, float)):
        raise ValueError(f"{namn}: sparad roc_auc måste vara numerisk")
    sparat_auc = float(sparat_varde)

    if not math.isfinite(sparat_auc) or not math.isclose(
        beraknat_auc, sparat_auc, rel_tol=0.0, abs_tol=METRIC_TOLERANCE
    ):
        raise ValueError(
            f"{namn}: omräknad ROC AUC ({beraknat_auc:.6f}) "
            f"matchar inte sparad ROC AUC ({sparat_auc:.6f})"
        )


def kontrollera_regression(
    reference_dir: Path = REFERENCE,
    outputs_dir: Path = OUTPUTS,
) -> bool:
    """Jämför referens och kandidat på samma dataversion och testuppsättning.

    joblib-filer kan exekvera kod vid inläsning: använd enbart betrodda artefakter.
    """

    referens = _las_metadata(reference_dir / "matvarden.json")
    kandidat = _las_metadata(outputs_dir / "matvarden.json")

    datafil = referens["datafil"]

    if not isinstance(datafil, str) or not datafil or Path(datafil).name != datafil:
        raise ValueError("Ogiltigt datafilnamn i referensens metadata")

    if kandidat["datafil"] != datafil:
        raise ValueError("Kandidat och referens använder olika datafiler")

    aktuell_hash = hashlib.sha256((DATAKATALOG / datafil).read_bytes()).hexdigest()

    if not (referens["data_sha256"] == kandidat["data_sha256"] == aktuell_hash):
        raise ValueError("Datahashen matchar inte: modellerna kan inte jämföras säkert")

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        champion = joblib.load(reference_dir / "modell.joblib")
        challenger = joblib.load(outputs_dir / "modell.joblib")

    _, X_test, _, y_test = hamta_tranings_och_testdata(datafil)
    champion_auc = float(roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1]))
    challenger_auc = float(roc_auc_score(y_test, challenger.predict_proba(X_test)[:, 1]))

    _kontrollera_sparat_auc("Champion", champion_auc, referens["roc_auc"])
    _kontrollera_sparat_auc("Challenger", challenger_auc, kandidat["roc_auc"])

    logger.info("Referenscommit: %s", referens.get("commit", "okänd"))
    logger.info("Data SHA256: %s", aktuell_hash)
    logger.info("Champion ROC AUC: %.6f", champion_auc)
    logger.info("Challenger ROC AUC: %.6f", challenger_auc)
    logger.info("Skillnad: %+.6f", challenger_auc - champion_auc)

    return godkann_modell(champion_auc, challenger_auc)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    try:
        godkand = kontrollera_regression()
    except (OSError, ValueError, KeyError, TypeError, IndexError, InconsistentVersionWarning):
        logger.exception("Kan inte genomföra modelljämförelsen")
        return 1

    if not godkand:
        logger.error("Challenger presterar sämre än Champion")
        return 1

    logger.info("Challenger klarar regressionskontrollen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
