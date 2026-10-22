import os
import subprocess
import sys

import numpy as np
import pandas as pd

import churn.score as score
from churn.score import main


def test_saknad_modellfil_stoppas(tmp_path, capsys):
    fil = tmp_path / "saknad_modell.joblib"

    resultat = main([str(fil)])
    utskrift = capsys.readouterr().out

    assert resultat == 1
    assert "FEL: hittar inte" in utskrift


def test_ringlista_prioriterar_ratt_kunder(tmp_path, monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return np.array(
                [
                    [0.20, 0.80],
                    [0.90, 0.10],
                    [0.40, 0.60],
                ]
            )

    kunder = pd.DataFrame(
        {
            "kund_id": [101, 102, 103, 104],
            "avslutat": [0, 1, 0, 0],
        }
    )

    monkeypatch.setattr(score.joblib, "load", lambda _: FakeModel())
    monkeypatch.setattr(score, "las_data", lambda: kunder)
    monkeypatch.setattr(score, "skapa_features", lambda df: df)
    monkeypatch.setattr(
        score,
        "dela_upp",
        lambda df: (df.drop(columns=["kund_id", "avslutat"]), None),
    )
    monkeypatch.setenv("ANTAL", "2")
    monkeypatch.chdir(tmp_path)

    modellfil = tmp_path / "modell.joblib"
    modellfil.touch()

    resultat = score.main([str(modellfil)])

    ringlista = pd.read_csv("outputs/ringlista.csv")

    assert resultat == 0
    assert ringlista["kund_id"].tolist() == [101, 104]
    assert ringlista["risk"].tolist() == [0.80, 0.60]


def test_cli_stoppar_vid_saknad_modell(tmp_path):
    fil = tmp_path / "saknad_modell.joblib"

    resultat = subprocess.run(
        [sys.executable, "-m", "churn.score", str(fil)],
        capture_output=True,
        text=True,
        env={**os.environ, "ANTAL": "2"},
        check=False,
    )

    assert resultat.returncode == 1
    assert "FEL: hittar inte" in resultat.stdout
