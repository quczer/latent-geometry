from typing import Callable, Iterable, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

import latent_geometry.viz.config as C


def path_to_trace(
    path: Callable[[float], np.ndarray],
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


def draw_spiders(
    spiders: list[list[Callable[[float], np.ndarray]]],
    background_trace: Optional[go.Scatter] = None,
) -> go.Figure:
    traces = []
    for spider in spiders:
        x0, y0 = spider[0](0.0)
        legend_group = f"spider_({x0:.1f},{y0:.1f})"
        traces.extend(
            path_to_trace(path, legend_group=legend_group, show_legend=i == 0)
            for i, path in enumerate(spider)
        )
    if background_trace:
        traces.append(background_trace)
    fig = plot_traces(traces)
    return fig


def plot_traces(traces: Iterable[BaseTraceType]) -> go.Figure:
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
