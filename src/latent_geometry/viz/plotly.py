from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

import latent_geometry.viz.config as C
from latent_geometry.path import Path


def create_background_trace(mus: np.ndarray, labels: np.ndarray) -> BaseTraceType:
    cmap = np.array(px.colors.qualitative.G10)
    colors = cmap[labels]
    return go.Scatter(
        x=mus[:, 0],
        y=mus[:, 1],
        mode="markers",
        marker=dict(color=colors, opacity=0.5),
        name="mnist",
    )


def draw_spiders(
    spiders: list[list[Path]],
    background_trace: Optional[go.Scatter] = None,
) -> go.Figure:
    traces: list[go.Scatter] = []
    for spider in spiders:
        x0, y0 = spider[0](0.0)
        legend_group = f"spider_({x0:.1f},{y0:.1f})"
        traces.extend(
            _path_to_trace(path, legend_group=legend_group, show_legend=i == 0)
            for i, path in enumerate(spider)
        )
    if background_trace:
        traces.append(background_trace)
    fig = plot_traces(traces)
    return fig


def plot_traces(traces: list[BaseTraceType]) -> go.Figure:
    fig = go.Figure(
        data=traces,
        layout={
            "width": C.FIGURE_WIDTH,
            "height": C.FIGURE_HEIGHT,
            "margin": {
                "b": C.MARGIN,
                "l": C.MARGIN,
                "r": C.MARGIN,
                "t": C.MARGIN,
            },
            "paper_bgcolor": "#fff",
            "yaxis": {"range": C.AXES_RANGE},
            "xaxis": {"range": C.AXES_RANGE},
        },
    )
    return fig


def _path_to_trace(
    path: Path,
    n_points: int = 30,
    color: str = "black",
    legend_group: Optional[str] = None,
    show_legend: bool = False,
) -> go.Scatter:
    timestamps = np.linspace(0.0, 1.0, n_points)
    points = np.vstack([path(t) for t in timestamps])
    return go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="lines",
        name=legend_group,
        line={"color": color, "width": C.LINE_WIDTH},
        opacity=C.LINE_OPACITY,
        legendgroup=legend_group,
        showlegend=show_legend,
    )
