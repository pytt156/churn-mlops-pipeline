import pandas as pd
import pandera.pandas as pa

from kontroller.schema import SCHEMA


def main(fil: str) -> int:
    try:
        df = pd.read_csv(fil)
    except FileNotFoundError:
        print(f"STOPP: {fil} saknas. Kontrollera filnamnet och sökvägen.")
        return 1
    except pd.errors.EmptyDataError:
        print(f"STOPP: {fil} är tom. Leverera en CSV med rubriker och datarader.")
        return 1
    except pd.errors.ParserError:
        print(f"STOPP: {fil} har ett oläsbart CSV-format. Kontrollera citattecken och avgränsare.")
        return 1
    except UnicodeDecodeError:
        print(f"STOPP: {fil} kunde inte avkodas. Spara CSV-filen med UTF-8-kodning.")
        return 1
    except OSError:
        print(f"STOPP: {fil} kunde inte läsas. Kontrollera filåtkomst och läsbehörighet.")
        return 1

    if df.empty:
        print(f"STOPP: {fil} innehåller inga datarader. Leverera en CSV med kundrader.")
        return 1

    try:
        SCHEMA.validate(df, lazy=True)

    except pa.errors.SchemaErrors as e:
        print(f"STOPP: {fil} klarar inte schemat")
        print(e.failure_cases)
        return 1

    print(f"OK: {fil} ({len(df)} rader)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1]))
