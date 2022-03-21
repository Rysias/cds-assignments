import networkx as nx
import pandas as pd
from typing import Dict, Callable

def df_to_graph(df: pd.DataFrame):
    return nx.from_pandas_edgelist(df, source="Source", target="Target", edge_attr=["Weight"])

def calc_measures(graph: nx.Graph, measure_dict: Dict[str, Callable]) -> pd.DataFrame:
    col_dict = {name: [] for name in measure_dict}
    for func_name, func in measure_dict.items():
        col_dict[func_name] = func(graph)
    return (
        pd.DataFrame.from_records(col_dict)
        .reset_index()
        .rename(columns={"index": "name"})
    )
