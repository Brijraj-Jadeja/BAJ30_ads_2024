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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

def run_k_means(input_frame : pandas.DataFrame, clusters):
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