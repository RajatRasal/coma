import os
from itertools import cycle, product
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyvista as pv
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pylab import get_cmap 

# pv.set_plot_theme("document")


def __format_facecolors(facecolors, cmap='YlGnBu'):
    cmap = get_cmap(cmap)
    facecolors = facecolors.copy()
    facecolors[facecolors < 0] = 0
    facecolors = cmap(facecolors)
    colorbar = plt.cm.ScalarMappable(cmap=cmap)
    return facecolors, colorbar

def __make_mesh_collection(mesh, faces, **kwargs):
    x = mesh[:, 0]
    y = mesh[:, 1]
    z = mesh[:, 2]

    tri = mtri.Triangulation(x, y, triangles=faces)

    _mesh = np.array([
        np.array([
            [x[T[0]], y[T[0]], z[T[0]]],
            [x[T[1]], y[T[1]], z[T[1]]],
            [x[T[2]], y[T[2]], z[T[2]]]
        ])
        for T in tri.triangles
    ])

    collection = Poly3DCollection(_mesh, **kwargs)
    return collection

def plot_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    figsize: Tuple[int, int] = (20, 15),
    elevations: List[int] = [0],
    azimuths: int = 5,
    alpha: float = 0.8,
    max_azimuth: int = 360,
    axis_scaling: List[int] = None,
    hide_grid: bool = False,
    show: bool = True,
    antialias: bool = False,
    edgecolor: str = 'gray',
    facecolors: np.ndarray = None,
    ax_lims: List[Tuple[int, int]] = None,
) -> Axes:
    xy = vertices[:, :2]
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=triangles)
    z = vertices[:, 2].flatten()

    if facecolors is not None:
        facecolors, colorbar = __format_facecolors(facecolors)

    nrows = len(elevations)
    ncols = azimuths
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw=dict(projection="3d"),
        constrained_layout=True,
    )
    ax = ax.reshape(-1, ncols)

    azimuth_intervals = max_azimuth / ncols
    elevation_intervals = 360 / nrows

    for j, elevation in enumerate(elevations):
        for i in range(ncols):
            azimuth = azimuth_intervals * i
            if axis_scaling is not None:
                # For anisotropic scaling of axis
                ax[j][i].set_box_aspect(axis_scaling)
            if hide_grid:
                ax[j][i].grid(False)
                ax[j][i].set_xticks([])
                ax[j][i].set_yticks([])
                ax[j][i].set_zticks([])
                ax[j][i].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax[j][i].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax[j][i].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax[j][i].set_axis_off()
            else:
                ax[j][i].set_title(f'E: {int(elevation)}, A: {int(azimuth)}')
            ax[j][i].view_init(elevation, azimuth)
            ax[j][i].set_xlabel('x')
            ax[j][i].set_ylabel('y')
            ax[j][i].set_zlabel('z')
            if facecolors is not None:
                ax[j][i].set_xlim3d(ax_lims[0][0], ax_lims[0][1])
                ax[j][i].set_ylim3d(ax_lims[1][0], ax_lims[1][1])
                ax[j][i].set_zlim3d(ax_lims[2][0], ax_lims[2][1])
                mesh_collection = __make_mesh_collection(
                    vertices, triangles,
                    facecolors=facecolors,
                    antialiaseds=antialias,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
                ax[j][i].add_collection3d(mesh_collection)
                # ax[j][i].autoscale(enable=True)
            else:
                if not antialias:
                    ax[j][i].plot_trisurf(triang, z, edgecolor='grey', alpha=alpha)
                else:
                    ax[j][i].plot_trisurf(triang, z, edgecolor=None, alpha=alpha,
                            antialiaseds=antialias)

    if facecolors is not None:
        fig.colorbar(colorbar, pad=0.01, shrink=0.2, location='right')

    if show:
        plt.show()
    else:
        return ax

    # V_T_list=None,
    # # if show:
    # #     plt.show() 
    # from scipy.spatial.transform import Rotation as R
    # # V_T = np.array([[-0.58081519,  0.34333078, -0.73809057],
    # #            [-0.68105154,  0.29170796,  0.67162137],
    # #                   [-0.44589518, -0.89276561, -0.06439754]]).T
    # r = R.from_rotvec([0, 0, np.pi * 0.5])
    # # V_T = r.apply(V_T)
    # center = np.array([
    #     [-104.429, 89.106, 67.9002],
    #     [-104.429, 89.106, 67.9002],
    #     [-104.429, 89.106, 67.9002],
    # ])

    # for row in ax:  #  in range(len(ax)):
    #     for _ax in row:
    #         for vt in V_T_list:
    #             V_T = r.apply(vt.T)
    #             _ax.quiver(center[:, 0], center[:, 1], center[:, 2],
    #                     V_T[0], V_T[1], V_T[2], # C=['green', 'blue', 'purple'],
    #                     length=40, lw=[10])

    # plt.show()

def plot_mesh_pyvista(
    plotter: pv.Plotter,
    polydata: pv.PolyData,
    # vertices: np.ndarray,
    # triangles: np.ndarray,
    figsize: Tuple[int, int] = (1024, 300),
    rotations: List[Tuple[int, int, int]] = [(0, 0, 0)],
    vertexcolors: List[int] = [],
    vertexscalar: str = '',
    cmap: str = 'YlGnBu',
    **mesh_kwargs,
):
    shape = plotter.shape
    assert shape[0] > 0 and shape[1] > 0
    assert shape[0] * shape[1] == len(rotations)

    if vertexscalar and vertexcolors:
        polydata[vertexscalar] = vertexcolors

    cmap = plt.cm.get_cmap(cmap)

    mesh_kwargs = {
        'cmap': cmap,
        'flip_scalars': True,
        'show_scalar_bar': False,
        **mesh_kwargs,
    }
    if vertexscalar and vertexcolors:
        mesh_kwargs['scalars'] = vertexscalar

    for subp, rots in zip(product(range(shape[0]), range(shape[1])), rotations):
        x, y, z = rots
        plotter.subplot(*subp)
        poly_copy = polydata.copy()
        poly_copy.rotate_x(x)
        poly_copy.rotate_y(y)
        poly_copy.rotate_z(z)
        plotter.add_mesh(
            poly_copy,
            **mesh_kwargs,
        )
        if rots == rotations[-1]:
            plotter.add_scalar_bar(label_font_size=10, position_x=0.85)


def plot_wireframe_and_meshes(
    vertices: np.ndarray,
    pred_verts: np.ndarray,
    triangles: np.ndarray,
    figsize: Tuple[int, int] = (20, 15),
    elevations: List[int] = [0],
    azimuths: int = 5,
    alpha: float = 0.8,
    wireframe_alpha: float = 0.0,
):
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles=triangles)
    z = vertices[:, 2].flatten()
    
    triang_pred = mtri.Triangulation(pred_verts[:, 0], pred_verts[:, 1], triangles=triangles)
    pred_z = pred_verts[:, 2].flatten()

    nrows = len(elevations)
    ncols = azimuths
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw=dict(projection="3d"),
    )
    ax = ax.reshape(-1, ncols)

    azimuth_intervals = 360 / ncols
    elevation_intervals = 360 / nrows

    for j, elevation in enumerate(elevations):
        for i in range(ncols):
            azimuth = azimuth_intervals * i
            ax[j][i].set_title(f'E: {int(elevation)}, A: {int(azimuth)}')
            ax[j][i].view_init(elevation, azimuth)
            ax[j][i].plot_trisurf(triang, z, edgecolor='grey', alpha=alpha)
            ax[j][i].plot_trisurf(triang_pred, pred_z, edgecolor='lightpink', alpha=wireframe_alpha)
            ax[j][i].set_xlabel('x')
            ax[j][i].set_ylabel('y')
            ax[j][i].set_zlabel('z')

    plt.show()

def plot_wireframes(
    vertices: np.ndarray,
    triangles: np.ndarray,
    figsize: Tuple[int, int] = (20, 15),
    elevations: List[int] = [0],
    azimuths: int = 5,
    alpha: float = 0.8,
    wireframe_alpha: float = 0.0,
    edge_colors=None,
    labels=None,
):
    """
    Plot wireframes on top of each other on the same plot
    """
    assert len(labels) == vertices.shape[0]

    nrows = len(elevations)
    ncols = azimuths
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw=dict(projection="3d"),
    )
    ax = ax.reshape(-1, ncols)

    azimuth_intervals = 360 / ncols
    elevation_intervals = 360 / nrows

    if edge_colors is None:
        edge_colors = cycle(['chocolate', 'orange', 'aquamarine', 'mediumpurple'])
    else:
        edge_colors = cycle(edge_colors)

    for j, elevation in enumerate(elevations):
        for i in range(ncols):
            azimuth = azimuth_intervals * i
            for v, label, color in zip(range(vertices.shape[0]), labels, edge_colors):
                vert = vertices[v]
                triang = mtri.Triangulation(vert[:, 0], vert[:, 1], triangles=triangles)
                z = vert[:, 2].flatten()
                p = ax[j][i].plot_trisurf(
                    triang, z,
                    edgecolor=color,
                    alpha=wireframe_alpha,
                )
                # y = p.get_label()
                # p.set_label('wdwdqd')
                # print(p.get_label())
                ax[j][i].set_title(f'E: {int(elevation)}, A: {int(azimuth)}')
                ax[j][i].view_init(elevation, azimuth)
                ax[j][i].set_xlabel('x')
                ax[j][i].set_ylabel('y')
                ax[j][i].set_zlabel('z')
                # ax[j][i].legend()
            # ax[j][i].set_label(['x', 'y', 'z'])
            # ax[j][i].legend()

    plt.show()

def vertex_moments(vertices: np.ndarray):
    batch_size = vertices.shape[0]
    vertices_reshaped = vertices.reshape(batch_size, -1)

    # global centering
    global_vertices_centered = vertices_reshaped - vertices_reshaped.mean(axis=0)
    
    # local centering
    local_vertices_centered = global_vertices_centered - \
        global_vertices_centered.mean(axis=1).reshape(batch_size, -1)

    U, S, V_T = np.linalg.svd(local_vertices_centered.T, full_matrices=True)
    return U, local_vertices_centered, global_vertices_centered

def plot_eigenmeshes(
    U: np.ndarray,
    n_modes: int,
    triangles: np.ndarray,
    figsize: Tuple[int, int] = (20, 15),
):
    nrows = int(np.ceil(n_modes / 5))
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=5,
        figsize=figsize,
        subplot_kw=dict(projection="3d"),
    )
    ax = ax.flatten()

    for i in range(n_modes):
        mode_i = U[i]
        triang = mtri.Triangulation(mode_i[:, 0], mode_i[:, 1], triangles=triangles)
        z = mode_i[:, 2].flatten()
        ax[i].set_title(f'Mode: {i}')
        ax[i].plot_trisurf(triang, z, edgecolor=None)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].set_zlabel('z')

    plt.show()
    
def plot_single_mesh(
    verts: np.ndarray,
    triangles: np.ndarray,
    title: str,
    figsize: Tuple[int, int] = (20, 15),
    wireframe: np.ndarray = None,
):
    alpha = 0.8
    wireframe_alpha = 0.2
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], triangles=triangles)
    z = verts[:, 2].flatten()
    ax.set_title(title)
    ax.plot_trisurf(triang, z, edgecolor='grey', alpha=alpha)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_mesh_grid(
    verts: np.ndarray,
    triangles: np.ndarray,
    titles: List[str],
    nrows: int = 4,
    ncols: int = 2,
    wireframe: np.ndarray = None,
    figsize: Tuple[int, int] = (30, 15),
    alpha: float = 0.8,
    img: bool = False,
):
    """
    Should be a batch of vertices (N, #verts, 3)
    """
    assert nrows * ncols == verts.shape[0]

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw=dict(projection="3d"),
    )
    # canvas = FigureCanvasAgg(fig)

    ax = ax.flatten()
    for i in range(nrows * ncols):
        triang = mtri.Triangulation(
            verts[i, :, 0],
            verts[i, :, 1],
            triangles=triangles
        )
        z = verts[i, :, 2].flatten()
        ax[i].plot_trisurf(triang, z, edgecolor='grey', alpha=alpha)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].set_zlabel('z')
        ax[i].set_title(titles[i])

    # if img:
    #     canvas.draw()
    #     buf = canvas.buffer_rgba()
    #     plot_img_array = np.asarray(buf)
    #     return plot_img_array
    # else:
    plt.show()


if __name__ == '__main__':
    x = plot_mesh_pyvista()
