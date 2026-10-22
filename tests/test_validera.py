import pandas as pd
import pytest

from kontroller.validera import main


@pytest.fixture
def csv_data():
    return pd.DataFrame(
        {
            "kund_id": [1, 2],
            "alder": [25, 40],
            "region": ["Stockholm", "Väst"],
            "manader_som_kund": [12, 24],
            "abonnemang": ["Bas", "Premium"],
            "manadskostnad": [199.0, 399.0],
            "data_gb_per_manad": [5.0, 20.0],
            "supportarenden_12m": [0, 2],
            "avslutat": [0, 1],
        }
    )


def test_giltig_csv_godkanns(tmp_path, capsys, csv_data):
    fil = tmp_path / "giltig.csv"
    csv_data.to_csv(fil, index=False)

    resultat = main(str(fil))

    utskrift = capsys.readouterr().out

    assert resultat == 0
    assert "OK:" in utskrift
    assert "2 rader" in utskrift


def test_ogiltig_csv_stoppas(tmp_path, capsys, csv_data):
    fil = tmp_path / "ogiltig.csv"

    csv_data.loc[0, "alder"] = 15
    csv_data.to_csv(fil, index=False)

    resultat = main(str(fil))
    utskrift = capsys.readouterr().out

    assert resultat == 1
    assert "STOPP:" in utskrift
    assert "alder" in utskrift


def test_saknad_csv_ger_fel(tmp_path):
    fil = tmp_path / "saknas.csv"

    with pytest.raises(FileNotFoundError):
        main(str(fil))
