import json
import os
import subprocess
import sys

import pytest

from churn.evaluate import main


def test_roc_auc_pa_gransen_godkanns(tmp_path, monkeypatch):
    monkeypatch.delenv("MIN_ROC_AUC", raising=False)

    fil = tmp_path / "matvarden.json"
    fil.write_text(json.dumps({"roc_auc": 0.70}))

    resultat = main([str(fil)])

    assert resultat == 0


def test_roc_auc_under_gransen_stoppas(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MIN_ROC_AUC", raising=False)

    fil = tmp_path / "matvarden.json"
    fil.write_text(json.dumps({"roc_auc": 0.69}))

    resultat = main([str(fil)])
    utskrift = capsys.readouterr().out

    assert resultat == 1
    assert "FEL:" in utskrift
    assert "under tröskeln" in utskrift


def test_anpassad_troskel(tmp_path, monkeypatch):
    monkeypatch.setenv("MIN_ROC_AUC", "0.85")

    fil = tmp_path / "matvarden.json"
    fil.write_text(json.dumps({"roc_auc": 0.80}))

    resultat = main([str(fil)])

    assert resultat == 1


def test_saknad_matvardesfil_stoppas(tmp_path, capsys):
    fil = tmp_path / "saknas.json"

    resultat = main([str(fil)])
    utskrift = capsys.readouterr().out

    assert resultat == 1
    assert "FEL: hittar inte" in utskrift


def test_ogiltig_json_ger_fel(tmp_path):
    fil = tmp_path / "matvarden.json"
    fil.write_text("{trasig json")

    with pytest.raises(json.JSONDecodeError):
        main([str(fil)])


def test_saknat_roc_auc_ger_fel(tmp_path):
    fil = tmp_path / "matvarden.json"
    fil.write_text(json.dumps({"accuracy": 0.92}))

    with pytest.raises(KeyError, match="roc_auc"):
        main([str(fil)])


def test_cli_stoppar_modell_under_troskel(tmp_path):
    fil = tmp_path / "matvarden.json"
    fil.write_text(json.dumps({"roc_auc": 0.50}))

    resultat = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn.evaluate",
            str(fil),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MIN_ROC_AUC": "0.70"},
        check=False,
    )

    assert resultat.returncode == 1
    assert "under tröskeln" in resultat.stdout
