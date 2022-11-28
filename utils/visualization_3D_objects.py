import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from IPython.display import clear_output

"""
visualization tools
"""

def getLayout(x, y, z, title = '3D Scatter plot'):

    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    max_range = max(x_range, max(y_range, z_range))

    x_min, x_max = np.min(x) - x_range*0.1, np.max(x) + x_range*0.1
    y_min, y_max = np.min(y) - y_range*0.1, np.max(y) + y_range*0.1
    z_min, z_max = np.min(z) - z_range*0.1, np.max(z) + z_range*0.1

    return go.Layout(title=title,
            scene = dict(
            xaxis = dict(range=[x_min, x_max],),
            yaxis = dict(range=[y_min, y_max],),
            zaxis = dict(range=[z_min, z_max],),
            aspectratio = dict(x=(x_max - x_min)/max_range*2,
                                y=(y_max - y_min)/max_range*2,
                                z=(z_max - z_min)/max_range*2)
            ),
            width = 1000,
            height = 700)

def getScatterTrace(X, point_size = 1.5, visible = True, color = None):
    color = X[1]**2 + X[2]**2 + 0.1*X[0]**2 if color == None else color
    return go.Scatter3d(
                visible=visible,
                x=X[0],
                y=X[1],
                z=X[2],
                mode='markers', marker=dict(
                    size=point_size,
                    color=color,  # set color to an array/list of desired values
                    colorscale='Viridis'
                )
            )

def draw3DPoints(X, point_size = 1.5, title = '3D Scatter plot', show = False):
    """
    X: np.ndarray (3, n) vertices
    """

    trace = getScatterTrace(X, point_size)

    layout = getLayout(X[0], X[1], X[2], title)

    fig = go.Figure(data=[trace], layout=layout)
    if show: fig.show()
    return fig

def save3DPointsImage(fig, camera = None, save_path = "image", title = "Scatter3D" ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if camera == None:      
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1, y=1, z=-2)
        )
    fig.update_layout(scene_camera=camera, title=title)
    fig.write_image(os.path.join(save_path, title + ".png"))

def saveList3DPointsImage(X_list, feat_idx, obj_idx, show = True):
    """
    X_list (np.ndarray): list of data to plot (N_step * N_obj * 3 * N_points)
    feat_idx (int): the index of the feature in feature space
    obj_idx (int): the index of the object to examine
    """

    for X in X_list:
        X = X[obj_idx]
        fig.add_trace(getScatterTrace(X, visible = False))

def draw3DMesh(X,F,title = '3D Scatter plot'):
    """
    X: np.ndarray (3, n) vertices
    F: np.ndarray (3, n) triangles
    """

    x,y,z = X
    i,j,k = F
    
    layout =  getLayout(x,y,z, title)

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,i=i,j=j,k=k, color='lightpink', opacity=0.50)], layout=layout)

    fig.show()

def save3DPoints(X):
    """
    X: np.ndarray (3, n) vertices
    """
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[0], X[1], X[2], marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig


# Create figure
def draw3DpointsSlider(X_list, feat_idx, obj_idx, show = True):
    """
    X_list (np.ndarray): list of data to plot (N_step * N_obj * 3 * N_points)
    feat_idx (int): the index of the feature in feature space
    obj_idx (int): the index of the object to examine
    """
    fig = go.Figure(layout = getLayout(X_list[:, :, 0], X_list[:, :, 1], X_list[:, :, 2], title = "Feature No.%d" % (feat_idx)))
    # Add traces, one for each slider step

    for X in X_list:
        X = X[obj_idx]
        fig.add_trace(getScatterTrace(X, visible = False))

    # Make 10th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []

    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=50,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)

    if show: fig.show()
    return fig

def comparePointClouds(X, Y, title = '3D Point Cloud', show = False):
    '''
    compare two point clouds
    @param X (np.ndarray(float, shape = (3, n)): input source point cloud
    @param Y (np.ndarray(float, shape = (3, n)): input target point cloud
    '''
    trace_X = go.Scatter3d(
        x=X[0], y=X[1], z=X[2], mode='markers', marker=dict(
            size=1,
            color="red",  # set color to an array/list of desired values
            colorscale='Viridis'
        ), name="source"
    )
    trace_Y = go.Scatter3d(
        x=Y[0], y=Y[1], z=Y[2], mode='markers', marker=dict(
            size=1,
            color="blue",  # set color to an array/list of desired values
            colorscale='Viridis'
        ), name="target"
    )
    layout = go.Layout(title=title)
    fig = go.Figure(data=[trace_X, trace_Y], layout=layout)
    if show: fig.show()
    return fig

def surfaceRecons(s_fit, alpha):
    """
    reconstruct the surface of the fit model
    alpha: parameter of alphaShape (supervised)
    """

    # transfer to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(s_fit.T)

    # # visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # print(f"alpha={alpha:.3f}")
    # print(mesh.compute_vertex_normals())

    # visualize the surface
    o3d.visualization.draw_geometries([mesh], window_name="car_fit",mesh_show_back_face=True)
    X, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    draw3DMesh(X, F, title='3D Scatter plot')