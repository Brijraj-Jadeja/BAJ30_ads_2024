from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

from . import assess
from . import address
import math
import requests
import zipfile
import csv
import time
import pymysql
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
#import httplib2
#import oauth2
#import tables
#import mongodb
#import sqlite
# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values,
or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights.
Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

# A relic of the past
def hello_world():
  print("Hello from the data science library!")


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def single_file_download(file_url : str, file_name : str, isZipped : bool = False) -> None:
    full_file_name = file_name + ".zip" if isZipped else ""
    print("Downloading", full_file_name)
    response = requests.get(file_url)

    if response.status_code == 200:
        print("Downloaded", full_file_name)
        with open("./" + full_file_name, "wb") as file:
            file.write(response.content)
        print("Saved", full_file_name)
        if isZipped:
            print("Unzipping", full_file_name)
            with zipfile.ZipFile("./" + full_file_name, "r") as zip_ref:
                zip_ref.printdir()
                zip_ref.extractall("./")
            print("Unzipped", full_file_name)
    else:
        print("Connection Failed")

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  print('Data stored for year: ' + str(year))
  conn.commit()

def generate_latitude_longitude_square_bounds(latitude : float, longitude : float, radius_distance_km : float = 1.0) -> tuple:
      #Make sure to half the box length when entering it as a radius
      dLat = radius_distance_km / 111.11
      dLong = dLat / math.cos(math.radians(latitude))
      north = latitude + dLat
      south = latitude - dLat
      east = longitude + dLong
      west = longitude - dLong
      return north, south, east, west

def database_box_query(conn, location : str,  lat_long : tuple, start_year : int, end_year : int,
                       radius : float = 1.0, csv_file_path : str = "output_file.csv"):
  north, south, east, west = generate_latitude_longitude_square_bounds(lat_long[0], lat_long[1], radius)
  north = str(north)
  south = str(south)
  east = str(east)
  west = str(west)

  start_date = str(start_year) + "-01-01"
  end_date = str(end_year) + "-12-31"
  cur = conn.cursor()


  print('Selecting data for provided location')

  view_name = location + str(start_year) +'_' + str(end_year)
  cur.execute('CREATE VIEW ' + view_name + ' AS SELECT * FROM prices_coordinates_data WHERE (date_of_transfer BETWEEN "'+ start_date +'" AND "' + end_date + '") AND (latitude BETWEEN ' + south + ' AND ' + north + ') AND (longitude BETWEEN ' + west + ' AND ' + east + ');')
  print("Created view")

  cur.fetchall()
  cur.execute(f'SELECT ' + view_name + '.*, pp_data.primary_addressable_object_name, pp_data.secondary_addressable_object_name, pp_data.street FROM ' + view_name + ' JOIN pp_data FORCE INDEX (idx_pp_date_transfer) ON ((pp_data.date_of_transfer = ' + view_name + '.date_of_transfer) AND ( pp_data.price = ' + view_name + '.price)) LIMIT 6000;')
  rows = cur.fetchall()

  cur.execute('DROP VIEW ' + view_name)
  print("Dropped view")

  cur.fetchall()
  print('Writing data for provided location')

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)

  conn.commit()

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    north, south, east, west = generate_latitude_longitude_square_bounds(latitude, longitude, distance_km)
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    pois_df = pd.DataFrame(pois)
    pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
    pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)
    poi_counts = {}
    for tag in tags.keys():
        if tag in pois_df.columns:
            poi_counts[tag] = pois_df[tag].notnull().sum()
        else:
            poi_counts[tag] = 0
    return poi_counts

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")

def download_data(url, folder_name, base_dir='', is_zipped = False):

    if os.path.exists(folder_name) and os.listdir(folder_name):
        print(f'Files already exist at: {folder_name}')
        return

    os.makedirs(folder_name, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    if is_zipped:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(folder_name)
    else:
        with open("response.jpg", "wb") as f:
            f.write(response.content)
    print(f'Files extracted to: {folder_name}')

def load_data(file_path):
    return pd.read_csv(file_path)

def load_census_data(code, level='oa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

def download_geo_data(base_dir=''):

  url = 'https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/6beafcfd9b9c4c9993a06b6b199d7e6d/geojson?layers=0'
  extract_dir = 'geo_data'

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with open(os.path.join(extract_dir, 'geo_data.geojson'), 'wb') as f:
    f.write(response.content)

  print(f"Files downloaded to: {extract_dir}")

def load_geo_data():
  return gpd.read_file(f'./geo_data/geo_data.geojson').set_index('OA21CD').sort_index()


def download_graphml(url, filename, pbf_flag = False):
    """
    Downloads a graphml file from a URL with a progress bar.
    """

    if os.path.exists(filename):
        print(f"File {filename} already exists.")
        return filename

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")

        if pbf_flag:
          print("Converting to graphml...")
          with osmium.SimpleWriter('united_kingdom_graph.osm') as  writer:
            for o in osmium.FileProcessor('united_kingdom_graph.osm.pbf'):
              writer.add(o)


        print(f"Download complete. File saved as {filename if not pbf_flag else filename[:-4]}")
        return filename if not pbf_flag else filename[:-4]
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None


def download_geo_data_csv(base_dir=''):

  url = 'https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/6beafcfd9b9c4c9993a06b6b199d7e6d/csv?layers=0'
  extract_dir = 'geo_data_csv'

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with open(os.path.join(extract_dir, 'geo_data.csv'), 'wb') as f:
    f.write(response.content)

  print(f"Files downloaded to: {extract_dir}")

def create_db_table(conn,table_name, db_field_type_pairs, db_name ='ads_2024'):
    curr = conn.cursor()
    use_query = f'USE {db_name};'
    curr.execute(use_query)
    drop_query = f"DROP TABLE IF EXISTS {table_name};"

    curr.execute(drop_query)
    create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    for db_field_type_pair in db_field_type_pairs:
        create_query += f"{db_field_type_pair[0]} {db_field_type_pair[1]}" + (" COLLATE utf8_bin" if db_field_type_pair[2] else "") + " NOT NULL, "

    create_query += "db_id bigint(20) unsigned NOT NULL "
    create_query = create_query + ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;"
    curr.execute(create_query)
    curr.execute(f'ALTER TABLE {table_name} ADD PRIMARY KEY (db_id);')
    curr.execute(f'ALTER TABLE {table_name} MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;')
    print(f"Table '{table_name}' created.")
    conn.commit()

def drop_db_table(conn, table_name):
  drop_query = f"DROP TABLE IF EXISTS {table_name};"
  curr = conn.cursor()
  curr.execute(drop_query)
  conn.commit()

def populate_db_table(conn, table_name, file_path, db_name ='ads_2024'):
  curr = conn.cursor()
  use_query = f'USE {db_name};'
  curr.execute(use_query)
  insert_query =  'LOAD DATA LOCAL INFILE "' + file_path + '" INTO TABLE ' + table_name + ' FIELDS TERMINATED BY ' + "','" +  ' OPTIONALLY ENCLOSED by ' + "'\"'" + ' LINES STARTING BY ' + "''" + ' TERMINATED BY ' + "'\\n'" + ';'
  curr.execute(insert_query)
  curr.execute(f'DELETE FROM {table_name} WHERE db_id = 1;')
  conn.commit()

def index_db_table(conn, table_name, index_name, index_columns, db_name = 'ads_2024'):
  curr = conn.cursor()
  use_query = f'USE {db_name};'
  curr.execute(use_query)
  index_query = f'CREATE INDEX {index_name} ON {table_name}({index_columns});'
  curr.execute(index_query)
  print(f"Index '{index_name}' created on '{table_name}' with columns '{index_columns}'.")
  conn.commit()

def query_db_table(conn, query, db_name = 'ads_2024'):
  curr = conn.cursor()
  use_query = f'USE {db_name};'
  curr.execute(use_query)
  curr.execute(query)
  output =  curr.fetchall()
  conn.commit()
  return output

def create_pd_database_connection(username, password, url, port):
  return sqlalchemy.create_engine(
    'mariadb+pymysql://{0}:{1}@{2}:{3}/{4}?local_infile=1'.format(username,
                                                                  password,
                                                                  url,
                                                                  str(port),
                                                                  'ads_2024')
    )


def upload_polygon_features(conn, geo_gdf, tags, table_name="output_area_features_data"):
    """
    Uploads features counts for each polygon in the GeoDataFrame to the database.

    Args:
        conn: Database connection object.
        geo_gdf: GeoDataFrame containing polygons.
        tags: Dictionary of OSM tags to count.
        table_name: Name of the database table to upload data to.
    """

    curr = conn.cursor()

    for index, row in tqdm.tqdm(geo_gdf.iterrows(), total=len(geo_gdf), desc="Processing Polygons"):
        try:
            polygon = row['geometry']
            features = count_features_in_polygon(tags, polygon)
            # Construct the SQL INSERT statement
            columns = ', '.join(features.keys())
            placeholders = ', '.join(['%s'] * (len(features)))
            values = [str(index)]
            values += list(features.values())
            sql = f"INSERT INTO {table_name} (geography_code, {columns}) VALUES (%s, {placeholders})"
            data = values # Include geography_code in the data
            # Execute the SQL statement
            curr.execute(sql, data)
            conn.commit()
        except Exception as e:
            print(f"Error processing polygon {index}: {e}")
            conn.rollback() # rollback transaction if any error occurs

    print(f"Features uploaded to table '{table_name}'.")
def query_db_table_to_dataframe(conn, query):
  """
    Queries a database

    Args:
        conn: Database connection object
        query
    """
    return pd.read_sql_query(query, con=conn)

def query_db_table_to_geodataframe(conn, query):
    df = pd.read_sql_query(query, con=conn)
    df['geometry'] = df['geometry'].apply(shapely.wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = 'EPSG:4326'
    return gdf


def download_data_from_gov(url, folder_name = None, headers = {}, base_dir='', is_zipped = False):
  """
    Downloads a file from a gov website

    Args:
        url : address of data
        folder_name :output folder name
        headers : HTTP get headers
        base_dir
        is_zipped
    """
  if is_zipped:
    extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
      print(f"Files already exist at: {extract_dir}.")
      return
    os.makedirs(extract_dir, exist_ok=True)
  else:
    file_name = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])
    os.makedirs(folder_name, exist_ok=True)
  session = requests.session()
  session.headers['User-Agent'] = 'Custom user agent'
  cook = { '__cf_bm' : '8N3isPX6tmagbyOJ00BKc2nqzzcwGK3vCk9ju.NQvhQ-1733266402-1.0.1.1-nGNIqh6ljdkVH3mpd9d1p_qJZt1jCIkLzkwX42wJA06YOzIKFSu_nrs_PUBDJPE0okkWqcbnFpcb0kiJiVKk0w'}
  response = session.get(url,headers=headers, cookies=cook)
  response.raise_for_status()

  if is_zipped:
      with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
          zip_ref.extractall(extract_dir)
      print(f'Files extracted to: {extract_dir}')
  else:
      with open(os.path.join(folder_name, file_name), 'wb') as f:
          f.write(response.content)
      print(f'Files extracted to: {folder_name}')