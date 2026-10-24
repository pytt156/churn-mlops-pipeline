"""Kontrollerar felkod och meddelande för olika CSV-leveranser."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJEKTROT = Path(__file__).resolve().parents[1]


def kor_validering(fil):
    return subprocess.run(
        [sys.executable, "-B", "-m", "kontroller.validera", str(fil)],
        cwd=PROJEKTROT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_giltig_kundfil_godkanns():
    fil = PROJEKTROT / "data" / "kundbas.csv"

    resultat = kor_validering(fil)

    assert resultat.returncode == 0
    assert f"OK: {fil}" in resultat.stdout
    assert "Traceback" not in resultat.stderr


@pytest.mark.parametrize(
    ("fall", "orsak", "vagledning"),
    [
        ("saknad", "saknas", "Kontrollera filnamnet och sökvägen"),
        ("tom", "är tom", "Leverera en CSV med rubriker och datarader"),
        ("bara_rubriker", "innehåller inga datarader", "Leverera en CSV med kundrader"),
        ("trasig", "oläsbart CSV-format", "Kontrollera citattecken och avgränsare"),
    ],
)
def test_felaktig_leverans_stoppas_med_tydligt_meddelande(tmp_path, fall, orsak, vagledning):
    fil = tmp_path / f"{fall}.csv"
    if fall == "tom":
        fil.write_text("", encoding="utf-8")
    elif fall == "bara_rubriker":
        with (PROJEKTROT / "data" / "kundbas.csv").open(encoding="utf-8") as kundfil:
            fil.write_text(kundfil.readline(), encoding="utf-8")
    elif fall == "trasig":
        fil.write_text('kund_id,alder\n100001,"75\n', encoding="utf-8")
    # I fallet "saknad" skapas ingen fil.

    resultat = kor_validering(fil)

    assert resultat.returncode == 1
    assert f"STOPP: {fil}" in resultat.stdout
    assert orsak in resultat.stdout
    assert vagledning in resultat.stdout
    assert "Traceback" not in resultat.stderr
