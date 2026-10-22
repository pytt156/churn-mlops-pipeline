"""Testnivå 3: modell. Kontrakt och reproducerbarhet."""

import hashlib
import json

import joblib
import numpy as np
import pytest

from churn import model
from churn.data import FEATURES, dela_upp, las_data, skapa_features
from churn.model import trana_och_utvardera


@pytest.fixture(scope="module")
def resultat():
    return trana_och_utvardera()


def test_modellkontrakt(resultat):
    modell, _ = resultat
    X, _ = dela_upp(skapa_features(las_data().head(5)))
    assert list(X.columns) == FEATURES
    sannolikhet = modell.predict_proba(X)
    assert sannolikhet.shape == (5, 2)
    assert np.all((sannolikhet >= 0) & (sannolikhet <= 1))


def test_reproducerbar(resultat):
    _, matvarden = resultat
    _, igen = trana_och_utvardera()
    assert igen == matvarden


def test_modell_roc_auc(resultat):
    _, matvarden = resultat

    assert matvarden["roc_auc"] >= 0.70


def test_main_sparar_artefakter(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "test-commit")

    model.main(utkatalog=tmp_path)

    modellfil = tmp_path / "modell.joblib"
    matvardesfil = tmp_path / "matvarden.json"

    assert modellfil.is_file()
    assert matvardesfil.is_file()

    sparad_modell = joblib.load(modellfil)
    matvarden = json.loads(matvardesfil.read_text())

    assert hasattr(sparad_modell, "predict_proba")
    assert matvarden["commit"] == "test-commit"
    assert matvarden["datafil"] == model.DATAFIL

    for nyckel in ("roc_auc", "accuracy"):
        assert 0.0 <= matvarden[nyckel] <= 1.0

    forvantad_hash = hashlib.sha256((model.DATAKATALOG / model.DATAFIL).read_bytes()).hexdigest()

    assert matvarden["data_sha256"] == forvantad_hash
