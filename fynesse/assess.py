from .config import *

from . import access
from . import address
import plotly.graph_objects as go
import pandas as pd
import csv

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

def heatmap(matrix, title="generic"):
    fig = px.imshow(matrix.mul(1),color_continuous_scale='Viridis_r',title="Location KM Distance Heatmap")
    fig.show()
def price_location_joiner(build_geo, csv_name):
  col_vals = ["Price", "Date", "Postcode", "Property_Type", "New_build", "Tenure", "locality",  "town_city",  "district",  "county",  "country",  "latitude",  "longitude","db_id", "primary_name", "secondary_name","street"]
  df = pd.read_csv(csv_name, names = col_vals, header=None, index_col=False)
  df['full_name'] = str(df['secondary_name'].fillna('') + df['primary_name'])
  df = df.drop(columns=["Property_Type", "New_build", "Tenure", "locality",  "town_city",  "district",  "county",  "country", "db_id"])

  build_geo = build_geo[build_geo['valid_address'] == True]
  build_geo['addr:street'] = build_geo['addr:street'].str.upper()

  output_df = pd.merge(df, build_geo,
                      left_on=['street', 'primary_name'],
                      right_on=['addr:street', 'addr:housenumber'],
                      how='inner')
    return output_df