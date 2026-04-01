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


def padel_court_2d_heatmap(x_data, y_data, colorscale="Hot", width=400):
    """
    Padel court with a 2D density heatmap overlay showing player position frequency.
    """
    fig = padel_court_2d(width=width)
    fig.add_trace(
        go.Histogram2dContour(
            x=x_data,
            y=y_data,
            colorscale=colorscale,
            reversescale=True,
            opacity=0.7,
            showscale=True,
            ncontours=20,
        )
    )
    return fig


def padel_court_2d_zones(width: int = 400):
    """
    Padel court with tactical zone boundary lines overlaid.

    Green dotted lines at |y| = 3 m  (front / transition boundary)
    Orange dotted lines at |y| = 6 m (transition / back-court boundary)
    """
    fig = padel_court_2d(width=width)

    # Front / transition boundary at |y| = 3
    for y_val in (-3, 3):
        fig.add_trace(
            go.Scatter(
                x=[-5, 5],
                y=[y_val, y_val],
                mode="lines",
                line=dict(color="rgba(0,200,100,0.7)", width=1, dash="dot"),
            )
        )

    # Transition / back-court boundary at |y| = 6
    for y_val in (-6, 6):
        fig.add_trace(
            go.Scatter(
                x=[-5, 5],
                y=[y_val, y_val],
                mode="lines",
                line=dict(color="rgba(255,140,0,0.7)", width=1, dash="dot"),
            )
        )

    return fig


def padel_court_heatmap_kde(
    x_series,
    y_series,
    player_half: str = "top",
    width: int = 380,
    title: str = "",
):
    """
    High-quality court position heatmap using a Gaussian KDE density estimate.

    Renders a green padel court with white lines as the background, then overlays
    a smooth density heatmap (transparent at low density → yellow → orange → deep red).

    Args:
        x_series:    pandas Series or array of player x positions (metres, -5 to +5)
        y_series:    pandas Series or array of player y positions (metres, -10 to +10)
        player_half: "top" if player's home half has y > 0; "bottom" if y < 0.
                     The player's own half is always drawn at the bottom of the chart
                     so the back wall is nearest to them.
        width:       figure width in pixels
        title:       optional chart title

    Returns:
        go.Figure
    """
    try:
        from scipy.stats import gaussian_kde
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    import pandas as pd

    height = int(width * 2.0)

    x = pd.to_numeric(pd.Series(x_series), errors="coerce").dropna().values
    y = pd.to_numeric(pd.Series(y_series), errors="coerce").dropna().values

    # Orient so player's half is at the bottom (positive y displayed inverted)
    # Court: y ∈ [-10, 10]. Players 1/2 are y>0; we flip so their back wall (y=10)
    # is at the bottom of the figure (display_y = -y).
    if player_half == "top":
        y_disp = -y
    else:
        y_disp = y

    fig = go.Figure()

    # ── Court background: filled green rectangle ─────────────────────────────
    fig.add_shape(type="rect", x0=-5, y0=-10, x1=5, y1=10,
                  fillcolor="#3a7d44", line_width=0, layer="below")

    # ── Court lines (white) ───────────────────────────────────────────────────
    def _line(x0, y0, x1, y1, width=2):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="white", width=width), layer="above")

    # Baselines and side walls
    _line(-5, -10, 5, -10, 3)
    _line(-5,  10, 5,  10, 3)
    _line(-5, -10, -5, 10, 3)
    _line( 5, -10,  5, 10, 3)
    # Net (thicker dashed)
    fig.add_shape(type="line", x0=-5, y0=0, x1=5, y1=0,
                  line=dict(color="white", width=3, dash="dot"), layer="above")
    # Service lines at ±7
    _line(-5, -7, 5, -7)
    _line(-5,  7, 5,  7)
    # Centre service line
    _line(0, -7, 0, 7)
    # Zone boundaries (subtle dotted)
    for yv in (-3, 3):
        fig.add_shape(type="line", x0=-5, y0=yv, x1=5, y1=yv,
                      line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
                      layer="above")
    for yv in (-6, 6):
        fig.add_shape(type="line", x0=-5, y0=yv, x1=5, y1=yv,
                      line=dict(color="rgba(255,220,100,0.4)", width=1, dash="dot"),
                      layer="above")

    # ── KDE density overlay ───────────────────────────────────────────────────
    if len(x) >= 5 and _has_scipy:
        xg = np.linspace(-5, 5, 60)
        yg = np.linspace(-10, 10, 120)
        xx, yy = np.meshgrid(xg, yg)
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Use display-oriented y for KDE so density follows visual orientation
        y_for_kde = y_disp if player_half == "top" else y_disp
        try:
            kde = gaussian_kde(np.vstack([x, y_disp]), bw_method=0.25)
            z = kde(positions).reshape(xx.shape)
        except Exception:
            z = None

        if z is not None:
            # Normalise 0→1, set very low values to NaN so they're transparent
            z_norm = z / z.max()
            z_norm[z_norm < 0.04] = np.nan

            # Custom colorscale: transparent → pale yellow → orange → deep red
            colorscale = [
                [0.0,  "rgba(255,255,204,0)"],
                [0.15, "rgba(255,255,153,0.55)"],
                [0.35, "rgba(254,178,76,0.75)"],
                [0.60, "rgba(240,59,32,0.88)"],
                [1.0,  "rgba(128,0,38,0.95)"],
            ]

            fig.add_trace(go.Heatmap(
                x=xg,
                y=yg,
                z=z_norm,
                colorscale=colorscale,
                zsmooth="best",
                showscale=False,
                hoverinfo="skip",
            ))
    elif len(x) >= 5:
        # Fallback: simple 2d histogram if scipy unavailable
        fig.add_trace(go.Histogram2dContour(
            x=x, y=y_disp,
            colorscale="Hot", reversescale=True,
            opacity=0.65, showscale=False, ncontours=12,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#e0e0e0"), x=0.5),
        paper_bgcolor="#1c1f27",
        plot_bgcolor="#1c1f27",
        xaxis=dict(
            range=[-5.3, 5.3], showgrid=False, zeroline=False,
            showticklabels=False, scaleanchor="y",
        ),
        yaxis=dict(
            range=[-10.5, 10.5], showgrid=False, zeroline=False,
            showticklabels=False,
        ),
        margin=dict(l=8, r=8, t=30 if title else 8, b=8),
        height=height,
        width=width,
        showlegend=False,
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