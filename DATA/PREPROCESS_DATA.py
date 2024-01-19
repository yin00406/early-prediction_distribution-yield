import sys
sys.path.append("../")
import os
import config
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from osgeo import gdal, gdalconst, osr
import scipy
from scipy.ndimage import label, generate_binary_structure
from skimage import morphology
import pandas as pd
from datetime import datetime

np.set_printoptions(precision=10,suppress=True,linewidth=3000)

cropName = "CS"
states = config.grids_SC[4:5] #c
years = config.year_list #c
doy_list = config.doys_CS

def data_validity():
    '''
    generate validation mask
    '''
    global doy_list
    print("Data Valid")
    for state in states:
        print(state)
        for year in years:
            print(year, end=" ")
            image_valid = []
            if year == 2020:
                doy_list = config.doys_CS_leap
            else:
                doy_list = config.doys_CS
            for doy in doy_list:
                print("DATE:{0}".format(doy), end=" ")
                raster = gdal.Open(os.path.join(config.DATA_DIR, state, state+"_"+str(year)+"_"+str(doy)+".tif"))
                band = raster.GetRasterBand(1)
                image_valid.append(~(np.isnan(band.ReadAsArray())))
            image_valid = np.array(image_valid, dtype=bool)
            np.save(os.path.join(config.NUMPY_DIR, state+"_"+str(year)+"_image_valid_temporal"), image_valid)

            image_valid = image_valid.all(axis=0)
            print("\nTile Size:{}".format(image_valid.shape))
            np.save(os.path.join(config.NUMPY_DIR, state+"_"+str(year)+"_image_valid"), image_valid)

def calculate_statistics():
    '''
    calculate statistics (max and min) for each region
    '''
    global doy_list
    print("Image Stats")
    for state in states:
        print(state)
        for year in years:
            image_stats = {}
            print(year, end=" ")
            if year == 2020:
                doy_list = config.doys_CS_leap
            else:
                doy_list = config.doys_CS
            for i,doy in enumerate(doy_list):
                print("DATE:{0}".format(doy), end=" ")
                doy = str(doy)
                raster = gdal.Open(os.path.join(config.DATA_DIR, state, state+"_"+str(year)+"_"+str(doy)+".tif"))
                image_stats[doy] = {}
                for channel in range(1, config.channels+1):
                    band = raster.GetRasterBand(channel)
                    if band.GetMinimum() is None or band.GetMaximum()is None:
                        band.ComputeStatistics(0)
                    print('\tMax:{}\tMin:{}\tNoData:{}\tBand:{}'.format(band.GetMaximum(), band.GetMinimum(), band.GetNoDataValue(), channel))
                    array = band.ReadAsArray()
                    image_stats[doy][channel] = {"Max": np.nanpercentile(array, 100-config.clip), "Min": np.nanpercentile(array, config.clip)}
                    print('\tMax:{}\tMin:{}\tClipped'.format(image_stats[doy][channel]["Max"], image_stats[doy][channel]["Min"]))
            with open(os.path.join(config.NUMPY_DIR, state+"_"+str(year)+"_image_stats"), 'wb') as f:
                pickle.dump(image_stats, f)

def create_preprocessed_images():
    '''
    Normalize images with recorded max and min
    '''
    global doy_list
    print("Process Images")
    for state in states:
        for year in years:
            print("State:{}".format(state))
            print("Data Valid")
            image_valid = np.load(os.path.join(config.NUMPY_DIR, state+"_"+str(year)+"_image_valid.npy"))
            # print(image_valid.shape)

            print("Image Stats")
            with open(os.path.join(config.NUMPY_DIR, state+"_"+str(year)+"_image_stats"), 'rb') as f:
                image_stats = pickle.load(f)

            if year == 2020:
                doy_list = config.doys_CS_leap
            else:
                doy_list = config.doys_CS
            image_full = np.zeros((len(doy_list), config.channels, image_valid.shape[0], image_valid.shape[1])).astype(np.float32)
            # print(state)
            for i,doy in enumerate(doy_list):
                doy = str(doy)
                raster = gdal.Open(
                    os.path.join(config.DATA_DIR, state, state + "_" + str(year) + "_" + str(doy) + ".tif"))
                for channel in range(1, config.channels+1):
                    band = raster.GetRasterBand(channel)
                    array = band.ReadAsArray()

                    array[array>image_stats[doy][channel]["Max"]] = image_stats[doy][channel]["Max"]
                    array[array<image_stats[doy][channel]["Min"]] = image_stats[doy][channel]["Min"]
                    if((image_stats[doy][channel]["Max"]-image_stats[doy][channel]["Min"]) != 0):
                        array = (array - image_stats[doy][channel]["Min"])/(image_stats[doy][channel]["Max"]-image_stats[doy][channel]["Min"])
                    else:
                        array = (array - image_stats[doy][channel]["Min"])

                    image_full[i, channel-1] = array.astype(np.float32)

            print(image_full.shape)
            np.save(os.path.join(config.NUMPY_DIR, state+"_"+str(year)+"_image"), image_full)

def convert_CDL_to_npy():
    '''
    convert CDL to npy data
    '''
    for state in states:
        for year in years:
            raster = gdal.Open(os.path.join(config.CDL_DIR, f"CDL_{year}_{state}_{cropName}.tif"))
            band = raster.GetRasterBand(1)
            array = band.ReadAsArray()
            np.save(os.path.join(config.NUMPY_DIR, f"CDL_{year}_{state}_{cropName}"), array)

if __name__ == "__main__":
    data_validity()
    calculate_statistics()
    create_preprocessed_images()
    convert_CDL_to_npy()