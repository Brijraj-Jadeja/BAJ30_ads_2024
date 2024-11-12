from .config import *

from . import access
from . import address
import plotly.graph_objects as go
import pandas as pd

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def df_to_radar_map(total_data_frame, scaled_total_array):
    categories = total_data_frame.columns[1:]
    scaled_total_df = pd.DataFrame(scaled_total_array, columns=categories)

    fig = go.Figure()

    for i in range(len(total_data_frame)):
    fig.add_trace(go.Scatterpolar(
            r=scaled_total_df.iloc[i],
            theta=categories,
            fill='toself',
            connectgaps=True,
            name = str(total_data_frame.iloc[i,0]) + ", " + str(total_data_frame.iloc[i,-1])
    ))
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[-2, 3]
        )),
    showlegend=True
    )

    fig.show()

import plotly.express as px
def distance_matrix_heatmap(distance_matrix):
    distance_matrix = pd.DataFrame(index=total_data_frame['Location'], columns=total_data_frame['Location'])

    for i in total_data_frame['Location']:
        for j in total_data_frame['Location']:
            distance_matrix.loc[i,j] = address.coordinate_distance_in_km(locations_dict[i], locations_dict[j])

        
    fig = px.imshow(distance_matrix.mul(1),color_continuous_scale='Viridis_r',title="Location KM Distance Heatmap")
    fig.show()