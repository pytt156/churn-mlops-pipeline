import pandas as pd
import pandera.pandas as pa
import pytest

from kontroller.schema import validate_data


@pytest.fixture
def giltig_df():
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


def test_giltig_data_godkanns(giltig_df):
    resultat = validate_data(giltig_df)

    pd.testing.assert_frame_equal(resultat, giltig_df)


def test_for_lag_alder_avvisas(giltig_df):
    giltig_df.loc[0, "alder"] = 17

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(giltig_df)


@pytest.mark.parametrize(
    "alder, ska_godkannas",
    [
        (17, False),
        (18, True),
        (110, True),
        (111, False),
    ],
)
def test_aldersgranser(giltig_df, alder, ska_godkannas):
    giltig_df.loc[0, "alder"] = alder

    if ska_godkannas:
        validate_data(giltig_df)
    else:
        with pytest.raises(pa.errors.SchemaErrors):
            validate_data(giltig_df)


def test_duplicerat_kund_id_avvisas(giltig_df):
    giltig_df.loc[1, "kund_id"] = 1

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(giltig_df)


def test_okand_region_avvisas(giltig_df):
    giltig_df.loc[0, "region"] = "Göteborg"

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(giltig_df)


def test_okant_abonnemang_avvisas(giltig_df):
    giltig_df.loc[0, "abonnemang"] = "Unlimited"

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(giltig_df)


def test_saknad_kolumn_avvisas(giltig_df):
    df = giltig_df.drop(columns=["manadskostnad"])

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(df)


def test_extra_kolumn_avvisas(giltig_df):
    giltig_df["kampanjkod"] = ["ABC", "XYZ"]

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(giltig_df)


@pytest.mark.parametrize(
    "kolumn, ogiltigt_varde",
    [
        ("manader_som_kund", -1),
        ("manadskostnad", 0.0),
        ("data_gb_per_manad", -0.1),
        ("supportarenden_12m", -1),
    ],
)
def test_ogiltiga_numeriska_varden_avvisas(giltig_df, kolumn, ogiltigt_varde):
    giltig_df.loc[0, kolumn] = ogiltigt_varde

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_data(giltig_df)

    assert kolumn in exc_info.value.failure_cases["column"].values


def test_flera_fel_rapporteras(giltig_df):
    giltig_df.loc[0, "alder"] = 17
    giltig_df.loc[1, "manadskostnad"] = 0.0

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_data(giltig_df)

    felkolumner = set(exc_info.value.failure_cases["column"].dropna())

    assert {"alder", "manadskostnad"} <= felkolumner
