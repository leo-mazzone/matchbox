import rustworkx as rx
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw

from matchbox.client._handler import get_resolution_graph
from matchbox.common.graph import ResolutionGraph


def draw_resolution_graph() -> Figure:
    """
    Draws the resolution graph.
    """
    res_graph = ResolutionGraph.model_validate_json(get_resolution_graph())
    G: rx.PyDiGraph = res_graph.to_rx()

    node_indices = G.node_indices()
    datasets = {
        G[node_indices[i]]["id"]: i
        for i in node_indices
        if G[node_indices[i]]["kind"] == "dataset"
    }

    colours = []
    for i in node_indices:
        kind = G[node_indices[i]]["kind"]
        if kind == "dataset":
            colours.append((0, 0, 1, 0.2))
        elif kind == "model":
            colours.append((1, 0, 0, 0.2))

    return mpl_draw(
        G,
        pos=rx.spring_layout(
            G,
            pos={v: [0, i / 2] for v, i in enumerate(datasets.values())},
            fixed=set(datasets.values()),
        ),
        node_color=colours,
        with_labels=True,
        labels=lambda node: node["name"],
        font_size=8,
    )
