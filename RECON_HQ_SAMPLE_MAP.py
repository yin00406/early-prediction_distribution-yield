import config
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import MODEL
import numpy as np
import os
import pickle
import torch
from osgeo import gdal, gdalconst, osr
import scipy
import copy
from collections import Counter
from skimage import morphology

print("#################### PARAMETER SETTING")
ROI = "E7"
model_ROI_list = ["NW", "SC", "SE"]
print("model_ROI_list: {}, ROI: {}".format(model_ROI_list, ROI))
year_list = [config.year_test, config.year_pred]
time_stops_preds = [6,9,12,15,18,21,24]
CA_suffix = "_TACA"
print("CA_suffix: {}".format(CA_suffix))
cropName = "CS"

map_sum_threshold_0 = 3
map_sum_threshold_2 = 3
map_sum_threshold_1 = 3
print("map_sum_threshold_0: {}".format(map_sum_threshold_0))
print("map_sum_threshold_1: {}".format(map_sum_threshold_1))
print("map_sum_threshold_2: {}".format(map_sum_threshold_2))
map_class_purity_threshold = 1
print("map_class_threshold: {}".format(map_class_purity_threshold))
SWITCH_EROSION = True
SWITCH_REMOVE_SMALL_OBJECTS = False

for model_ROI in model_ROI_list:
    print("model_ROI: ", model_ROI)
    for year in year_list:
        print("year: ", year)
        for time_stops_pred in time_stops_preds:
            print("time stop: ", time_stops_pred)
            data_array_CDL = np.load(os.path.join(config.NUMPY_DIR, f"CDL_{year}_{ROI}_{cropName}.npy"))
            map_count_all = np.zeros((3, data_array_CDL.shape[0],data_array_CDL.shape[1]))
            for itr in range(20):
                with open(os.path.join(config.MAP_COUNT_DIR,
                                       f"{ROI}_{model_ROI}_{time_stops_pred}_{config.input_patch_size}_{year}{CA_suffix}_{cropName}_MAP_COUNT_{itr}"
                                       ), 'rb') as f:
                    map_count = pickle.load(f)
                print(f"LOAD COUNT MAP: {itr}")
                map_count_all += map_count

            map_sum = np.sum(map_count_all, axis=0)
            map_max = np.max(map_count_all, axis=0)
            map_probability = map_max/map_sum
            map_argmax = np.argmax(map_count_all,axis=0)

            # ASSIGN (0,0,0) TO NAN
            map_argmax[np.isnan(map_probability)] = config.nanValue

            # DIFFERENT map_corn_sum_threshold AND map_soybean_sum_threshold TO MASK OUT INVALID PIXELS
            map_others_mask = (map_argmax ==0)
            map_corn_mask = (map_argmax==1)
            map_soybean_mask = (map_argmax==2)
            map_sum_threshold_mask_0 = (map_sum < map_sum_threshold_0)
            map_sum_threshold_mask_1 = (map_sum < map_sum_threshold_1)
            map_sum_threshold_mask_2 = (map_sum < map_sum_threshold_2)
            map_argmax[map_others_mask&map_sum_threshold_mask_0] = config.nanValue
            map_argmax[map_corn_mask&map_sum_threshold_mask_1] = config.nanValue
            map_argmax[map_soybean_mask&map_sum_threshold_mask_2] = config.nanValue

            # ONLY RETAIN PIXELS WITH HIGH PURITY
            map_argmax[map_probability<map_class_purity_threshold] = config.nanValue

            # EROSION
            if SWITCH_EROSION == True:
                eroded_label = config.nanValue * np.ones_like(map_argmax)
                for class_val in range(len(config.classes)):
                    binary_label_class = np.zeros_like(eroded_label)
                    binary_label_class[map_argmax == class_val] = 1
                    binary_label_class = scipy.ndimage.binary_erosion(binary_label_class, iterations=config.iterations)
                    eroded_label[binary_label_class == True] = class_val
                map_argmax = copy.deepcopy(eroded_label)
                for key, value in sorted(Counter(np.reshape(map_argmax, (-1))).items()):
                    print("{} : {}".format(key, value))

            if SWITCH_REMOVE_SMALL_OBJECTS == True:
                # REMOVE SMALL OBJECTS
                label_valid = (map_argmax != config.nanValue)
                connected_components_left = morphology.remove_small_objects(label_valid.astype(bool), config.small_objects_threshold)
                map_argmax[connected_components_left == False] = config.nanValue
                for key, value in sorted(Counter(np.reshape(map_argmax, (-1))).items()):
                    print("{} : {}".format(key, value))

            # CHECK CONFUSION MATRIX
            print(classification_report(data_array_CDL.flatten(), map_argmax.flatten(), digits=6))

            # SAVE MAP
            tif_with_meta = gdal.Open(os.path.join(config.CDL_DIR, f"CDL_{year}_{ROI}_{cropName}.tif"), gdalconst.GA_ReadOnly)
            filename = os.path.join(config.PRED_MAP, f"{ROI}_{model_ROI}_{time_stops_pred}_{year}{CA_suffix}_{cropName}.tif")
            gt = tif_with_meta.GetGeoTransform()
            driver = gdal.GetDriverByName("GTiff")
            dest = driver.Create(filename, map_argmax.shape[1], map_argmax.shape[0], 1, gdal.GDT_Byte, options = ["COMPRESS=DEFLATE"])
            dest.GetRasterBand(1).WriteArray(map_argmax)
            dest.SetGeoTransform(gt)
            wkt = tif_with_meta.GetProjection()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
            dest.SetProjection(srs.ExportToWkt())
            dest = None