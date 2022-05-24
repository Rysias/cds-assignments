import networkx as nx  # type: ignore
import src.calc_measures as cm


def test_calc_measures():
    graph = nx.watts_strogatz_graph(10, 3, 0.3)
    measure_dict = {
        "degree_centrality": nx.degree_centrality,
        "betweenness_centrality": nx.betweenness_centrality,
    }
    measure_df = cm.calc_measures(graph, measure_dict)
    assert len(measure_df.columns) == 3
    assert "name" in measure_df.columns
    assert measure_df["degree_centrality"].min() >= 0
    assert measure_df["betweenness_centrality"].min() >= 0
