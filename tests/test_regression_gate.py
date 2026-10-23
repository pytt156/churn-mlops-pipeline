import json
import shutil

import pytest
from sklearn.metrics import roc_auc_score

import churn.regression_gate as gate


@pytest.fixture
def artifacts(tmp_path):
    """Skapa isolerade kopior av godkända modellartefakter."""
    reference = tmp_path / "reference"
    outputs = tmp_path / "outputs"
    reference.mkdir()
    outputs.mkdir()

    for name in ("modell.joblib", "matvarden.json"):
        shutil.copy2(gate.REFERENCE / name, reference / name)
        shutil.copy2(gate.REFERENCE / name, outputs / name)

    return reference, outputs


@pytest.mark.parametrize(
    ("champion", "challenger", "expected"),
    [
        (0.78, 0.82, True),
        (0.78, 0.78, True),
        (0.78, 0.77, False),
        (0.78, 0.69, False),
        (float("nan"), 0.80, False),
        (0.78, float("inf"), False),
    ],
)
def test_beslutsregel(champion, challenger, expected):
    assert gate.godkann_modell(champion, challenger) is expected


def test_likvardiga_modeller_godkanns(artifacts):
    reference, outputs = artifacts

    assert gate.kontrollera_regression(reference, outputs) is True


def test_forsamrad_modell_stoppas(artifacts, monkeypatch):
    """En modell med omvänd rangordning ska underkännas."""
    reference, outputs = artifacts
    original_load = gate.joblib.load

    class ForsamradModell:
        def __init__(self, model):
            self.model = model

        def predict_proba(self, X):
            return self.model.predict_proba(X)[:, ::-1]

    champion = original_load(reference / "modell.joblib")
    challenger = ForsamradModell(champion)

    _, X_test, _, y_test = gate.hamta_tranings_och_testdata()
    challenger_auc = float(roc_auc_score(y_test, challenger.predict_proba(X_test)[:, 1]))

    metadata_file = outputs / "matvarden.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["roc_auc"] = challenger_auc
    metadata_file.write_text(json.dumps(metadata))

    def load_model(path):
        if path == outputs / "modell.joblib":
            return challenger
        return original_load(path)

    monkeypatch.setattr(gate.joblib, "load", load_model)

    assert gate.kontrollera_regression(reference, outputs) is False


def test_felaktig_datahash_stoppas(artifacts):
    reference, outputs = artifacts
    metadata_file = outputs / "matvarden.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["data_sha256"] = "felaktig-hash"
    metadata_file.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Datahashen"):
        gate.kontrollera_regression(reference, outputs)


def test_felaktigt_sparat_auc_stoppas(artifacts):
    reference, outputs = artifacts
    metadata_file = outputs / "matvarden.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["roc_auc"] = 0.95
    metadata_file.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="matchar inte sparad"):
        gate.kontrollera_regression(reference, outputs)


def test_saknad_referensmodell_stoppas(artifacts):
    reference, outputs = artifacts
    (reference / "modell.joblib").unlink()

    with pytest.raises(FileNotFoundError):
        gate.kontrollera_regression(reference, outputs)


def test_main_returnerar_felkod_vid_regression(monkeypatch):
    monkeypatch.setattr(gate, "kontrollera_regression", lambda: False)

    assert gate.main() == 1


def test_main_returnerar_felkod_vid_fel(monkeypatch):
    def raise_error():
        raise ValueError("Ogiltig referens")

    monkeypatch.setattr(gate, "kontrollera_regression", raise_error)

    assert gate.main() == 1


def test_main_returnerar_noll_vid_godkand_modell(monkeypatch):
    monkeypatch.setattr(gate, "kontrollera_regression", lambda: True)

    assert gate.main() == 0
