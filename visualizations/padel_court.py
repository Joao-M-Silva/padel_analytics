import plotly.graph_objects as go


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