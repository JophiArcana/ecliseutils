"""Small matplotlib plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib import transforms
from matplotlib.patches import Ellipse


__all__ = ["color", "confidence_ellipse"]


def color(z: float, scale: float = 120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
    """Plot the covariance confidence ellipse of *x* and *y* onto ``ax``.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radii.
    **kwargs
        Forwarded to ``matplotlib.patches.Ellipse``.
    """
    x, y = np.array(x), np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    M = np.stack([x, y], axis=0)
    cov = (M @ M.T) / len(x)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
