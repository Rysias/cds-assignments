import argparse
import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
from spacy.tokens import Doc
from pathlib import Path
from spacytextblob.spacytextblob import SpacyTextBlob
from typing import Sequence, Callable, List, Tuple

NLP = spacy.load("en_core_web_sm")
NLP.add_pipe('spacytextblob')

def extract_geopol(doc: Doc) -> str:
    return ";".join(ent.text for ent in doc.ents if ent.label_ == "GPE")

def list_geopol(docs: Sequence[Doc]) -> List[str]:
    return [extract_geopol(doc) for doc in docs]

def textblob_sentiment(doc: Doc) -> float:
    return doc._.blob.polarity

def list_sentiment(docs: Sequence[Doc], sent_f: Callable[[Doc], float]) -> List[float]:
    return [sent_f(doc) for doc in docs]

def process_df(df: pd.DataFrame, sent_fun=textblob_sentiment) -> pd.DataFrame:
    headline_docs = list(NLP.pipe(df["title"]))
    geopols = list_geopol(headline_docs)
    sentiments = list_sentiment(headline_docs, sent_fun)
    return pd.DataFrame(zip(df["title"], geopols, sentiments), columns = ["title", "GPE", "sentiment"])
    
def flatten_series(series: pd.Series) -> pd.Series:
    return series.apply(pd.Series).stack().reset_index(drop=True)

def split_entities(ents: pd.Series) -> List[str]:
    non_empty_ents = ents[ents.str.len() > 0]
    split_ents = non_empty_ents.str.split(";")
    return flatten_series(split_ents)

def most_common_ents(ents: pd.Series, n=20) -> pd.DataFrame:
    ent_series = split_entities(ents)
    return ent_series.value_counts()[:n].rename_axis('Entity').reset_index(name='Count')

def plot_top_ents(ent_df, output_dir, top_n=20, group_type="Real"):
    top_ents = most_common_ents(ent_df["GPE"], n=top_n)
    plot_title = f"Most Mentioned {group_type} News GPEs"
    sns.barplot(data=top_ents, y="Entity", x="Count", orient="h", color="#29C5F6").set(title=plot_title)
    plt.savefig(str(output_dir / f"{group_type}_top_ents.png"))

def read_news(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=0).reset_index(drop=True)
    
def create_output_dir() -> Path:
    output_dir = Path("output/")
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir

    
def main(args):
    topn = args.top_n
    DATA_PATH = Path(args.data_dir)
    
    news_df = read_news(DATA_PATH)
    mask = df["label"] == "REAL"
    real_df = df[mask]
    fake_df = df[~mask]
    
    top_ents_real = process_df(real_df)
    top_ents_fake = process_df(fake_df)
    
    # Write output
    output_dir = create_output_dir()
    top_ents_real.to_csv(output_dir / "Real_GPE_sent.csv")        
    top_ents_fake.to_csv(output_dir / "Fake_GPE_sent.csv")
    plot_top_ents(top_ents_real, output_dir, group_type="Real")
    plot_top_ents(top_ents_fake, output_dir, group_type="Fake")
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Finds Geopolitical entities (GEOP) and their sentiment from news headlines from a dataset containing fake and real news. Also plots the top n (default 20) most mentioned entities"
    )
    argparser.add_argument(
        "--top-n",
        default=20,
        type=int,
        help="Number of entities to plot (default: 20)",
    )
    argparser.add_argument(
        "--data-path",
        default="../../../CDS-LANG/tabular_examples/fake_or_real_news.csv",
        type=int,
        help="Path to the fake_or_real_news.csv file",
    )
    args = argparser.parse_args()
    main(args)
