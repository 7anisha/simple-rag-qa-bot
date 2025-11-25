import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SRC = DATA_DIR / "wikipedia_movie_plots.csv"

def load_dataset(path=SRC):
    df = pd.read_csv(path)
    # expected columns: 'Title', 'Plot', 'Year', ...
    df = df.fillna("")
    df["doc_id"] = df.index.astype(str)
    # create a single text field
    df["text"] = df["Plot"].astype(str)
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    print(df.head(2).to_dict(orient="records"))
