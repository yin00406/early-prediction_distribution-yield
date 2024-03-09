import config
import os
import numpy as np
import DATA_LOADER
import pickle
from osgeo import gdal, gdalconst

switch = "test"
print("switch: {}".format(switch))
if switch == "train":
    generate_all_save = True
    combine_all_save = False

ROI_list = config.grids_SE[1:]
region = "D4" # train
year_list = [2019,2020,2021] # train
CLS_year_list = [config.year_test, config.year_pred]
year_list_pred = [config.year_test, config.year_pred]
time_stops = range(6,25)
CLS_time_stops = [21,24]
time_stops_pred =  [6,9,12,15,18,21,24]
cropName = "CS"
CA_suffix = "_TACA"

thresholds_year_stop = {0: 1024 * 3, 1: 1024 * 3, 2: 1024 * 3}
crop_labels = [0,1,2]

if not os.path.exists(os.path.join(config.NUMPY_DIR, "dataset")):
    os.makedirs(os.path.join(config.NUMPY_DIR, "dataset"))

if switch == "train":
    if generate_all_save == True:
        for ROI in ROI_list:
            for year in year_list:
                for time_stop in time_stops:
                    print("#######################################################################")
                    print("ROI: {}\nyear: {}\ntime_stop: {}".format(ROI, year, time_stop))

                    print('loading train data')
                    train_data_array = np.load(os.path.join(config.NUMPY_DIR, f"{ROI}_{year}_image.npy"))[:time_stop]
                    print(f"train_data_array.shape: {train_data_array.shape}")
                    train_data_valid_array_temporal = np.load(
                        os.path.join(config.NUMPY_DIR, f"{ROI}_{year}_image_valid_temporal.npy"))[:time_stop]
                    print(f"train_data_valid_array_temporal.shape: {train_data_valid_array_temporal.shape}")
                    label_array = np.load(os.path.join(config.NUMPY_DIR, f"CDL_{year}_{ROI}_{cropName}.npy"))
                    print(f"label_array.shape: {label_array.shape}")

                    print('creating train patches')
                    train_image_patches, label_ids = DATA_LOADER.create_patches_train_random(train_data_array,
                                                                                             train_data_valid_array_temporal,
                                                                                             label_array,
                                                                                             crop_labels,
                                                                                             thresholds_year_stop,
                                                                                             time_stop,
                                                                                             config.patch_purity_threshold,
                                                                                             config.input_patch_size)

                    print(np.array(label_ids).shape)
                    print(f"The number of valid patches for each class: {np.bincount(np.array(label_ids))}")

                    data_train = DATA_LOADER.SEGMENTATION_train(train_image_patches, label_ids)
                    # if combine_all_save_stop == True:
                    with open(os.path.join(config.NUMPY_DIR, "dataset",
                                           f"{ROI}_{year}_{time_stop}_{config.input_patch_size}_trainDataset_{cropName}"),
                              'wb') as f:
                        pickle.dump(data_train, f)

    if combine_all_save == True:
        combined_image_patches = []
        combined_label_ids = []
        for ROI in ROI_list:
            for year in year_list:
                for time_stop in time_stops:
                    print("ROI: {}\nyear: {}\ntime_stop: {}".format(ROI, year, time_stop))
                    with open(os.path.join(config.NUMPY_DIR, "dataset",
                                           f"{ROI}_{year}_{time_stop}_{config.input_patch_size}_trainDataset_{cropName}"),
                              'rb') as f:
                        dataset = pickle.load(f)
                    combined_image_patches += dataset.image_patches
                    combined_label_ids += dataset.label_ids

        data_train = DATA_LOADER.SEGMENTATION_train(combined_image_patches, combined_label_ids)
        with open(os.path.join(config.NUMPY_DIR, "dataset", "year_stop",
                               f"{region}_{config.input_patch_size}_trainDataset_{cropName}"), 'wb') as f:
            pickle.dump(data_train, f)

if switch == "test":
    print("#######################################################################")
    for ROI in ROI_list:
        for year in year_list_pred[:]:
            for time_stop in time_stops_pred[:]:
                print("ROI: {}\nyear: {}\ntime_stop: {}".format(ROI, year, time_stop))
                print('loading test data')
                test_data_array = np.load(os.path.join(config.NUMPY_DIR, f"{ROI}_{year}_image.npy"))[:time_stop]
                print(f"test_data_array.shape: {test_data_array.shape}")
                test_data_valid_array_temporal = np.load(
                    os.path.join(config.NUMPY_DIR, f"{ROI}_{year}_image_valid_temporal.npy"))[:time_stop]
                print(f"test_data_valid_array_temporal.shape: {test_data_valid_array_temporal.shape}")

                print('creating test patches')
                for itr in range(20):
                    no_patches_to_create = 200000
                    test_image_patches, image_height_starts, image_width_starts = DATA_LOADER.create_patches_test(
                        test_data_array,test_data_valid_array_temporal,no_patches_to_create, time_stop)
                    
                    print(f"The number of patches: {len(test_image_patches)}")
                    data_test = DATA_LOADER.SEGMENTATION_test(test_image_patches, image_height_starts, image_width_starts)
                    with open(os.path.join(config.NUMPY_DIR, "dataset", "predict",
                                           f"{ROI}_{year}_{time_stop}_{config.input_patch_size}_testDataset_{cropName}_{itr+1}"),
                              'wb') as f:
                        pickle.dump(data_test, f)