# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""
from . import access
from . import assess
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import osmnx as ox
import fynesse
import requests
import zipfile
import io
import os
import pandas as pd
import geopandas as gpd
import yaml
import shapely
import sqlalchemy
import tqdm
import osmium
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ipywidgets import interact_manual, Text, Password

def plot_k_means_feature_inertia(total_data_frame):
    total_dataframe_to_scale = total_data_frame.loc[:, total_data_frame.columns != 'Location']
    scaled_total_array = StandardScaler().fit_transform(total_dataframe_to_scale)
    sse = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=12112024)
        kmeans.fit(scaled_total_array)
        sse.append(kmeans.inertia_)

    #visualize results
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

def run_k_means(input_frame : pd.DataFrame, clusters):
    frame_to_scale = input_frame.loc[:, input_frame.columns != 'Location']
    scaled_frame_array = StandardScaler().fit_transform(frame_to_scale)
    kmeans = KMeans(n_clusters=4, random_state=12112024)
    y = kmeans.fit_predict(scaled_frame_array)
    output_frame = input_frame.copy()
    output_frame['Cluster'] = y
    return output_frame


def coordinate_distance_in_km(lat_long_1, lat_long_2):
  #Use Haversine
  R = 6371
  lat1,long1 = lat_long_1
  lat2,long2 = lat_long_2
  dlat = abs(math.radians(lat1 - lat2))
  dlong = abs(math.radians(long1 - long2))
  lat1 = math.radians(lat1)
  lat2 = math.radians(lat2)

  hav_theta = math.pow(math.sin(dlat/2),2) + math.pow(math.sin(dlong/2),2) * math.cos(lat1) * math.cos(lat2)
  d = 2 * math.asin(math.sqrt(hav_theta))
  return R * d

def polygon_code_from_coordinate(polygon_gdf, longitude, latitude, code_column=None):
  point = shapely.Point(longitude, latitude)
  index = polygon_gdf.sindex.query(point,predicate="within")
  if len(index) == 0:
    return None
  if code_column is not None:
    return polygon_gdf[code_column][index[0]]
  return index[0]

def tag_filter(valid_tags, given_tags):
  output_set = set()
  for i in given_tags.keys():
    if i in valid_tags and given_tags[i] != 'no':
      output_set.add(i)
      if given_tags[i] in valid_tags:
        output_set.add(given_tags[i])
  return output_set

def add_tags_to_gdf(gdf,index, tags):
  for i in tags:
    if i not in gdf['tags'][index]:
      gdf.loc[index, 'tags'][i] = 1
    else:
      gdf.loc[index, 'tags'][i] += 1

def count_features_in_polygon(tags: dict, polygon, file_path=None) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        tags (dict): A dictionary of OSM tags to filter the features (e.g., {'amenity': True, 'tourism': True}).
        polygon (shapely.geometry.Polygon): The polygon to search within.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    if file_path is None:
      features_gdf = ox.features_from_polygon(polygon, tags)
    else:
      features_gdf = ox.features.features_from_xml(file_path,polygon,tags)
    return features_gdf
    feature_counts = {}
    for tag in tags.keys():
        if tag in features_gdf.columns:
            feature_counts[tag] = features_gdf[tag].notnull().sum()
        else:
            feature_counts[tag] = 0
    return feature_counts


def format_gdf_tags_for_database(input_gdf):
  input_gdf_local = input_gdf.reset_index(inplace=False)
  output_gdf = input_gdf_local[['OA21CD']].copy()
  output_gdf.rename(columns={'OA21CD': 'geography_code'}, inplace=True)
  output_gdf
  high_level_tags = set(['amenity', 'barrier', 'building', 'highway', 'emergency', 'manmade', 'public_transport', 'railway', 'waterway'])

  amenity = [0 for i in range(len(output_gdf))]
  barrier = [0 for i in range(len(output_gdf))]
  building = [0 for i in range(len(output_gdf))]
  highway = [0 for i in range(len(output_gdf))]
  emergency = [0 for i in range(len(output_gdf))]
  manmade = [0 for i in range(len(output_gdf))]
  public_transport = [0 for i in range(len(output_gdf))]
  railway = [0 for i in range(len(output_gdf))]
  waterway = [0 for i in range(len(output_gdf))]
  tags = [{} for i in range(len(output_gdf))]

  output_gdf['amenity'] = amenity
  output_gdf['barrier'] = barrier
  output_gdf['building'] = building
  output_gdf['highway'] = highway
  output_gdf['emergency'] = emergency
  output_gdf['manmade'] = manmade
  output_gdf['public_transport'] = public_transport
  output_gdf['railway'] = railway
  output_gdf['waterway'] = railway
  output_gdf['tags'] = tags
  for index, row in tqdm.tqdm(input_gdf_local.iterrows(), total=len(input_gdf_local)):
    current_tags = row['tags']
    for i in current_tags.keys():
      if i in high_level_tags:
        output_gdf.loc[index, i] = current_tags[i]
      else:
        output_gdf.loc[index, 'tags'][i] = current_tags[i]
    output_gdf.loc[index, 'tags'] = str(output_gdf.loc[index, 'tags'])
  return output_gdf


def extract_dictionary_tags(given_frame):
  training_df = given_frame.iloc[:, 1:-3]
  existing_tags = []
  seen_tags = set()
  #The added complexity should give determinism i think

  for index, row in tqdm.tqdm(given_frame.iterrows(), total=len(given_frame),desc='Generating new columns'):
      for tag in eval(row['tags']).keys():
        if tag not in seen_tags:
          existing_tags.append(tag)
          seen_tags.add(tag)

  augmented_training_df = training_df.copy()
  for i in existing_tags:
    augmented_training_df[i] = [0 for i in range(len(training_df))]

  for index, row in tqdm.tqdm(given_frame.iterrows(), total=len(given_frame),desc = 'populating columns'):
    temp_dict = eval(row['tags'])
    for tag in temp_dict.keys():
      augmented_training_df.loc[index, tag] = temp_dict[tag]

  return augmented_training_df


def produce_linear_model(augmented_training_df, answer_df, answer_col_name,print_results=True,print_chart=True,print_corr=True,seed=2122024):
  print("Producing linear model")
  np.random.seed(2122024)

  y = answer_df[[answer_col_name]]


  x_aug = augmented_training_df.to_numpy()
  y = y.to_numpy()

  x_aug = sm.add_constant(x_aug)

  design = x_aug

  m_linear_aug = sm.OLS(y, design)
  results_basis_aug = m_linear_aug.fit()
  print("Fitting Model")

  y_pred_linear_aug = results_basis_aug.get_prediction(design).summary_frame(alpha=0.05)
  y = pd.Series(y.flatten())

  if print_results:
    print(results_basis_aug.summary())
  if print_chart:
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred_linear_aug['mean'], alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Correlation between Predicted and Actual Values")

    plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red')

    plt.show()
  if print_corr:
    correlation = pd.Series(y).corr(pd.Series(y_pred_linear_aug['mean']))
    print(correlation)
  return results_basis_aug.params