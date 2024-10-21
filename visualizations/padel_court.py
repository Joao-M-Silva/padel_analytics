import io
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import  moviepy.editor as mpy


def padel_court_2d(
    width: int = 400,
):
    """
    Padel court 
    """
    height = width * 2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-5, 5], 
            y=[-10, -10],
            mode='lines',
            line=dict(
                color="gray",
                width=8,
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-5, 5], 
            y=[-7, -7],
            mode='lines',
            line=dict(
                color="gray",
                width=2,
            ),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[-5, 5], 
            y=[0,0],
            mode='lines',
            line=dict(
                color="gray",
                width=2,
                dash="dash",
            ),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[-5, 5], 
            y=[7, 7],
            mode='lines',
            line=dict(
                color="gray",
                width=2,
            ),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[-5, 5], 
            y=[10, 10],
            mode='lines',
            line=dict(
                color="gray",
                width=8,
            ),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[-5, -5], 
            y=[-10, 10],
            mode='lines',
            line=dict(
                color="gray",
                width=8,
            ),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0], 
            y=[-7, 7],
            mode='lines',
            line=dict(
                color="gray",
                width=2,
            ),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[5, 5], 
            y=[-10, 10],
            mode='lines',
            line=dict(
                color="gray",
                width=8,
            ),
        ),
    )
    
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            title="Base Line",
            tick0=-5, 
            dtick=1,
            range=[-5, 5]
    
        ),
        yaxis=dict(
            showgrid=False,
            title="Side Line",
            tick0=-10,
            dtick=2,
        ),
        showlegend=False,
        height=height,
        width=width,    
    )

    return fig


def plotly_fig2array(fig):
    """
    Convert a plotly figure to numpy array
    """
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)
"""
n = 20 # number of radii
h = 2/(n-1)
r = np.linspace(h, 2,  n)
theta = np.linspace(0, 2*np.pi, 60)
r, theta = np.meshgrid(r,theta)
r = r.flatten()
theta = theta.flatten()

x = r*np.cos(theta)
y = r*np.sin(theta)

# Triangulate the circular  planar region
tri = Delaunay(np.vstack([x,y]).T)
faces = np.asarray(tri.simplices)
I, J, K = faces.T

f = lambda h: np.sinc(x**2+y**2)+np.sin(x+h)   

fig = go.Figure(go.Mesh3d(x=x,
                     y=y,
                     z=f(0),
                     intensity=f(0),
                     i=I,
                     j=J,
                     k=K,
                     colorscale='matter_r', 
                     showscale=False))
                     
fig.update_layout(title_text='My hat is flying with MoviePy',
                  title_x=0.5,
                  width=500, height=500, 
                  scene_xaxis_visible=False, 
                  scene_yaxis_visible=False, 
                  scene_zaxis_visible=False)

# No Plotly frames are defined here!! Instead we define moviepy frames by
# converting each Plotly figure to  an array, from which MoviePy creates a clip
# The concatenated clips are saved as a gif file:
def make_frame(t):
    z = f(2*np.pi*t/2)
    fig.update_traces(z=z, intensity=z)  #These are the updates that usually are performed within Plotly go.Frame definition
    return plotly_fig2array(fig)

animation = mpy.VideoClip(make_frame, duration=2)
animation.write_gif("image/my_hat.gif", fps=20)
"""