"""Write the Iris dataset to data/raw/iris.csv.

The output is deterministic, so the file always has the same hash. CI uses this
as a fallback when the DVC remote is not reachable (e.g. a local remote).
"""
from pathlib import Path

from sklearn.datasets import load_iris

OUT = Path("data/raw/iris.csv")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df.drop(columns="target").to_csv(OUT, index=False, float_format="%.1f", lineterminator="\n")
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
