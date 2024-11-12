from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

import math
import requests
import zipfile
import csv
import time
import pymysql
import pandas as pd
import osmnx as ox
import math
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