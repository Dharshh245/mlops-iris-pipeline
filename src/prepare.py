"""Stage 1 - prepare: split raw data into train/test sets."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/iris.csv")
OUT_DIR = Path("data/prepared")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    df = pd.read_csv(RAW)

    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["species"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"train={len(train)} rows, test={len(test)} rows -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
