from cartographer import *
import numpy as np
from plotly.graph_objs import *
import plotly.plotly as py
import plotly


# Creates a Z-order curve layout. Used for visualization.
def z_layout(n, two_dimensional=True):
    pos = {}
    for i in range(0, n):
        if two_dimensional:
            pos[i] = (*deinterleave(i), 0)
        else:
            pos[i] = deinterleave3d(i)
    return pos

# Creates a Symmetrical Z-order curve layout. Used for visualization
def sym_z_layout(n, two_dimensional=True):
    pos = {}
    for i in range(0, n):
        if two_dimensional:
            pos[i] = (*deinterleave(half_twirl(i)), 0)
        else:
            pos[i] = deinterleave3d(half_twirl3d(i))
    return pos

def visualize(phenotype, two_dimensional, symmetric):

    graph = phenotype

    Edges = []
    group = []
    labels = set()

    N = np.size(graph, 0)

    for row in range(N):
      for col in range(N):
        if graph[ row, col ] != 0:
          Edges.append((row, col))
          group.append(graph[row,col])
          group.append(graph[row,col])
          group.append(graph[row,col])
          labels.add(row)
          labels.add(col)

    if symmetric:
        layout = sym_z_layout(N, two_dimensional)
    else:
        layout = z_layout(N, two_dimensional)

    Xn=[]
    Yn=[]
    Zn=[]
    for label in labels:
        Xn.append(layout[label][0])
        Yn.append(layout[label][1])
        Zn.append(layout[label][2])

    labels = list(map(str, labels))

    nodes = 'rgb(255, 255, 255)'

    Xe=[]
    Ye=[]
    Ze=[]
    for e in Edges:
        Xe+=[layout[e[0]][0],layout[e[1]][0], None]
        Ye+=[layout[e[0]][1],layout[e[1]][1], None]
        Ze+=[layout[e[0]][2],layout[e[1]][2], None]


    trace1=Scatter3d(
                   x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=Line(color=group, colorscale='Viridis', width=group, 
                   cmin=-3, cmax=3, reversescale=True),
                   surfacecolor=group,
                   hoverinfo='none'
                   )
    trace2=Scatter3d(
                   x=Xn,
                   y=Yn,
                   z=Zn,
                   mode='markers',
                   name='actors',
                   marker=Marker(symbol='dot',
                                 size=2,
                                 color=nodes,
                                 colorscale='Greys',
                                 cmin=0,
                                 cmax=3,
                                 line=Line(color='rgb(50,50,50)', width=0.5)
                                 ),
                   text=labels,
                   hoverinfo='text'
                   )

    axis = dict(
              showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              showaxeslabels=False,
              title=''
              )

    camera = dict(
              up=dict(x=0, y=0, z=1),
              center=dict(x=0, y=0, z=0),
              eye=dict(x=0.0, y=-2.0, z=1.0) 
              )

    layout = Layout(
             title="Phenotype",
             width=1000,
             height=1000,
             showlegend=True,
             paper_bgcolor='rgb(0,0,0)',
             scene=Scene(
             xaxis=XAxis(axis),
             yaxis=YAxis(axis),
             zaxis=ZAxis(axis),
             camera = camera
            ),
            margin=Margin(t=100),
            hovermode='closest',
            annotations=Annotations([
              Annotation(
              showarrow=False,
              text=" ",
              xref='paper',
              yref='paper',
              x=0,
              y=0.1,
              xanchor='left',
              yanchor='bottom',
              font=Font(size=14)
              )]),    
        )

    data=Data([trace1, trace2])
    fig=Figure(data=data, layout=layout)

    #plotly.offline.plot
    py.plot({
    	"data": data,
    	"layout": layout
    	})


#graph = np.eye(512, k=1)
#graph = np.zeros((8,8))
#graph[0][2] = 1
#graph[4][6] = 1
#graph = np.kron(graph, graph)
#graph = np.kron(graph, graph)
#symmetric = True
#two_dimensional = False

#visualize(graph, two_dimensional, symmetric)