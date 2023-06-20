from typing import Callable

import numpy as np
import plotly.graph_objects as go

import sys

import sys
sys.path.append("C:\\Users\j.rutkowski2\\PycharmProjects\\HydrAMP\\latent-geometry\\src")

from latent_geometry.manifold import Manifold
from latent_geometry.visual.calc import get_circles, get_lines, get_geodesics
from latent_geometry.visual.config import (
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    LINE_OPACITY,
    LINE_WIDTH,
)


def create_scatter_object_given_path(
    path: Callable[[float], np.ndarray], n_points: int = 30
) -> go.Scatter:
    timestamps = np.linspace(0.0, 1.0, n_points)
    points = np.vstack([path(t) for t in timestamps])
    return go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="lines",
        name="",
        line={"color": "black", "width": LINE_WIDTH},
        opacity=LINE_OPACITY,
    )


def create_topology_fig(
    centre: np.ndarray,
    mu: np.ndarray,
    log_var: np.ndarray,
    manifold: Manifold,
    background_trace: go.Scatter,
    num_lines: int,
    num_circles: int,
    line_length: float = 2.5,
    show_lines: bool = True,
    show_circles: bool = True,
) -> go.Figure:
    lines = get_lines(centre, mu, log_var, num_lines, manifold, length=line_length)
    circles = get_circles(lines, num_circles)

    traces = [background_trace]
    if show_lines:
        traces.extend([create_scatter_object_given_path(line) for line in lines])
    if show_circles:
        traces.extend([create_scatter_object_given_path(circle) for circle in circles])

    fig = go.Figure(layout={"width": FIGURE_WIDTH, "height": FIGURE_HEIGHT})
    for trace in traces:
        fig.add_trace(trace)
    return fig



def create_topology_fig_geodesics(
    centers_1: np.ndarray,
    centers_2,
    manifold: Manifold,
    background_trace: go.Scatter,
    # num_lines: int,
    # num_circles: int,
    # line_length: float = 2.5,
    # show_lines: bool = True,
    # show_circles: bool = True,
) -> go.Figure:
    geodesics = get_geodesics(centers_1, centers_2, manifold)
    # lines = get_lines(center, num_lines, manifold, length=line_length)
    # circles = get_circles(lines, num_circles)

    traces = [background_trace]
    # if show_lines:
    #     traces.extend([create_scatter_object_given_path(line) for line in lines])
    # if show_circles:
    #     traces.extend([create_scatter_object_given_path(circle) for circle in circles])

    traces.extend([create_scatter_object_given_path(geodesic) for geodesic in geodesics])
    fig = go.Figure(layout={"width": FIGURE_WIDTH, "height": FIGURE_HEIGHT})
    for trace in traces:
        fig.add_trace(trace)
    return fig
