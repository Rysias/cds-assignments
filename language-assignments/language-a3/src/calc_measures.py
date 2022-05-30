"""
Functions for calculating measures in the network
"""
import networkx as nx
import pandas as pd
from typing import Dict, Callable


def df_to_graph(df: pd.DataFrame):
    return nx.from_pandas_edgelist(
        df, source="Source", target="Target", edge_attr=["Weight"]
    )


def calc_measures(graph: nx.Graph, measure_dict: Dict[str, Callable]) -> pd.DataFrame:
    col_dict = {name: [] for name in measure_dict}
    for func_name, func in measure_dict.items():
        col_dict[func_name] = func(graph)
    return (
        pd.DataFrame.from_records(col_dict)
        .reset_index()
        .rename(columns={"index": "name"})
    )


def add_measures(graph: nx.Graph, measure_dict: Dict[str, Callable]) -> None:
    """Modify graph with measures from measure_dict"""
    for func_name, func in measure_dict.items():
        nx.set_node_attributes(graph, func(graph), func_name)


def network_to_df(graph: nx.Graph) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        .reset_index()
        .rename(columns={"index": "name"})
    )

