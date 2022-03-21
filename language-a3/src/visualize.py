from pathlib import Path
from networkx import Graph  # type: ignore
from pyvis.network import Network  # type: ignore


def plot_graph(graph: Graph, filename: Path):
    """Saves graph of network using standard parameters """
    output_path = Path("output") / f"{filename.stem}_viz.html"
    nt = Network()
    nt.from_nx(graph)
    nt.save_graph(str(output_path))
