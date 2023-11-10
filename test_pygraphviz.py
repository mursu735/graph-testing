import pygraphviz as pgv
import plotly.express as px
import numpy as np

G = pgv.AGraph()

G.add_node("a")  # adds node 'a'

G.add_edge("b", "c")  # adds edge 'b'-'c' (and also nodes 'b', 'c')

