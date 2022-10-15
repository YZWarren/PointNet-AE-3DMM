import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

"""
visualization tools
"""

def draw3DPoints(X, title = '3D Scatter plot'):
    """
    X: np.ndarray (3, n) vertices
    """

    x_range = np.max(X[0]) - np.min(X[0])
    y_range = np.max(X[1]) - np.min(X[1])
    z_range = np.max(X[2]) - np.min(X[2])
    max_range = max(x_range, max(y_range, z_range))

    trace = go.Scatter3d(
        x=X[0], y=X[1], z=X[2], mode='markers', marker=dict(
            size=1,
            color='blue',  # set color to an array/list of desired values
            colorscale='Viridis'
        )
    )
    layout = go.Layout(title=title,
                       scene = dict(
                           xaxis = dict(range=[np.min(X[0]) - x_range*0.1, np.max(X[0]) + x_range*0.1],),
                           yaxis = dict(range=[np.min(X[1]) - y_range*0.1, np.max(X[1]) + y_range*0.1],),
                           zaxis = dict(range=[np.min(X[2]) - z_range*0.1, np.max(X[2]) + z_range*0.1],),
                           aspectratio = dict(x=((np.max(X[0]) + x_range*0.1) - (np.min(X[0]) - x_range*0.1))/max_range,
                                              y=((np.max(X[1]) + y_range*0.1) - (np.min(X[1]) - y_range*0.1))/max_range,
                                              z=((np.max(X[2]) + z_range*0.1) - (np.min(X[2]) - z_range*0.1))/max_range)
                       ),
                       width = 1000,
                       height = 700)
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def draw3DMesh(X,F,title = '3D Scatter plot'):
    """
    X: np.ndarray (3, n) vertices
    F: np.ndarray (3, n) triangles
    """

    x_range = np.max(X[0]) - np.min(X[0])
    y_range = np.max(X[1]) - np.min(X[1])
    z_range = np.max(X[2]) - np.min(X[2])
    max_range = max(x_range, max(y_range, z_range))

    x,y,z = X
    i,j,k = F
    

    layout = go.Layout(title=title,
                       scene = dict(
                           xaxis = dict(range=[np.min(X[0]) - x_range*0.1, np.max(X[0]) + x_range*0.1],),
                           yaxis = dict(range=[np.min(X[1]) - y_range*0.1, np.max(X[1]) + y_range*0.1],),
                           zaxis = dict(range=[np.min(X[2]) - z_range*0.1, np.max(X[2]) + z_range*0.1],),
                           aspectratio = dict(x=((np.max(X[0]) + x_range*0.1) - (np.min(X[0]) - x_range*0.1))/max_range,
                                              y=((np.max(X[1]) + y_range*0.1) - (np.min(X[1]) - y_range*0.1))/max_range,
                                              z=((np.max(X[2]) + z_range*0.1) - (np.min(X[2]) - z_range*0.1))/max_range)
                       ),
                       width = 1000,
                       height = 700)

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


def comparePointClouds(X, Y, title = '3D Point Cloud'):
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
    fig.show()

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