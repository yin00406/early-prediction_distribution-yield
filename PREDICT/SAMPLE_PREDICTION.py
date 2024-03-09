import config
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import MODEL
import numpy as np
import os
import pickle
import torch
from scipy.spatial import distance
from osgeo import gdal, gdalconst, osr
import gc
import random
import DATA_LOADER

# CONFIGURATION
ROI = "D5"
model_ROI_list = ["SC"]
kmeans_training = False
config.batch_size = 1024*3
time_stops_preds = [6,9,12,15,18,21,24] # for prediction
cropName = "CS"
year_list = [config.year_test, config.year_pred] #

print("#################### PARAMETER SETTING")
num_clusters = 500
print("num_clusters: {}".format(num_clusters))
CA_suffix = "_TACA"
print("CA_suffix: {}".format(CA_suffix))
epoches = 5000
# print("epoches: {}".format(epoches))
purity_threshold = 0.95 # percentile: 0-1
print("purity_threshold: {}".format(purity_threshold))
label_count_threshold = 150 # number
print("label_count_threshold: {}".format(label_count_threshold))
boundary_radius_threshold = 95 # percentile: 0-100
print("boundary_radius_threshold: {}".format(boundary_radius_threshold))

print("#################### BUILD MODEL")
model = getattr(MODEL, "UNET_LSTM_BIDIRECTIONAL_AUTOENCODER" + CA_suffix + "_4")(in_channels=config.channels,
                                                                          out_channels=config.channels)
model = model.to('cuda')
for model_ROI in model_ROI_list:
    print("model_ROI: {}".format(model_ROI), "ROI: {}".format(ROI))
    modelLoaded = f"best_{model_ROI}_{cropName}{CA_suffix}_150.pt"
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, modelLoaded)),
                          strict=False)
    print(f"modelLoaded: {modelLoaded}")

    print("#################### LOAD TRAIN DATA")
    with open(os.path.join(config.NUMPY_DIR, "dataset/year_stop", f"{model_ROI}_{config.input_patch_size}_trainDataset_{cropName}"),
              'rb') as f:
        data_train = pickle.load(f)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=config.batch_size, shuffle=False,
                                                    num_workers=0, generator=torch.Generator(device='cuda'))

    print("#################### GET TRAIN IMAGES EMBEDDINGS AND LABELS")
    train_reps = []
    train_labels = []
    for batch, [image_patch, label_batch] in enumerate(data_loader_train):
        image_patch[torch.isnan(image_patch)] = 0
        code_vec, out_recon, _, _ = model(image_patch.to('cuda').float())

        for b in range(code_vec.shape[0]):
            train_reps.append(code_vec[b].detach().cpu().numpy())
            train_labels.append(label_batch[b].detach().cpu().numpy())

    train_labels_array = np.array(train_labels)

    if kmeans_training == True:
        print("#################### CLUSTERING WITH TRAIN DATA")
        kmeans_all_reps = MiniBatchKMeans(n_clusters=num_clusters, max_iter=epoches, batch_size=config.batch_size * 10).fit(
            train_reps)
        with open(os.path.join(config.MAP_COUNT_DIR, f"{model_ROI}_kmeans"),
                  'wb') as f:
            pickle.dump(kmeans_all_reps, f)

        raise Exception("debug")

    else:
        with open(os.path.join(config.MAP_COUNT_DIR, f"{model_ROI}_kmeans"),
                  'rb') as f:
            kmeans_all_reps = pickle.load(f)

    for year in year_list:
        print("YEAR: {}".format(year))
        # if year == 2022:
        #     time_stops_preds = time_stops_preds[-1:]
        # else:
        #     time_stops_preds = [6,9,12,15,18,21,24]
        for time_stops_pred in time_stops_preds[:]:
            print("time stop: ", time_stops_pred)
            for itr in range(20): # start from 0
                with open(os.path.join(config.NUMPY_DIR, "dataset/predict", f"{ROI}_{year}_{time_stops_pred}_{config.input_patch_size}_testDataset_{cropName}_{itr+1}"), 'rb') as f:
                    data_test = pickle.load(f)
                data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=config.batch_size, shuffle=False, num_workers=0, generator=torch.Generator(device='cuda'))

                print("#################### GET TEST IMAGES EMBEDDINGS AND LABELS")
                test_reps = []
                test_height_starts = []
                test_width_starts = []
                for batch, [image_patch, image_height_starts_batch, image_width_starts_batch] in enumerate(data_loader_test):

                    image_patch[torch.isnan(image_patch)] = 0
                    code_vec, out_recon, _, _ = model(image_patch.to('cuda').float())
                    for b in range(code_vec.shape[0]):
                        test_reps.append(code_vec[b].detach().cpu().numpy())
                        test_height_starts.append(image_height_starts_batch[b].detach().cpu().numpy())
                        test_width_starts.append(image_width_starts_batch[b].detach().cpu().numpy())
                print('# of test patches encoded: ', len(test_reps))

                print("#################### CLUSTERING AND ASSIGNMENT")
                train_rep_pred = kmeans_all_reps.predict(train_reps)
                # print("train_rep_pred.shape: {}".format(train_rep_pred.shape))
                transform_train_matrix = kmeans_all_reps.transform(train_reps) #
                # print("transform_train_matrix.shape: {}".format(transform_train_matrix.shape))
                boundary_radius_npy = np.zeros(num_clusters)
                cluster_label = np.zeros(num_clusters).astype(np.int8) + 100
                # print("cluster_label:{}".format(cluster_label))

                for cluster_No in range(num_clusters):

                    count_cluster = np.sum(train_rep_pred == cluster_No)
                    # print(count_cluster)

                    if (count_cluster != 0):
                        subset_labels = train_labels_array[train_rep_pred == cluster_No]
                        subset_transform_matrix = transform_train_matrix[train_rep_pred == cluster_No] # only select embeddings classified into this cluster
                        # print("subset_transform_matrix.shape: {}".format(subset_transform_matrix.shape)) # subset_transform_matrix.shape: (134, 500)
                        subset_transform_matrix_cluster = subset_transform_matrix[:, cluster_No] # only select the distance of this cluster
                        # print("subset_transform_matrix_cluster.shape: {}".format(subset_transform_matrix_cluster.shape))
                        label_cluster, sum_cluster = np.argmax(np.bincount(subset_labels)), np.sum(np.bincount(subset_labels))
                        cluster_label_count = np.bincount(subset_labels)[label_cluster]
                        percentage_max = cluster_label_count / sum_cluster
                        # print(cluster_No, np.bincount(subset_labels), label_cluster, sum_cluster, cluster_label_count, percentage_max)
                        if (percentage_max > purity_threshold) & (cluster_label_count > label_count_threshold):

                            subset_transform_matrix_cluster_label = subset_transform_matrix_cluster[subset_labels == label_cluster]
                            boundary_radius = np.percentile(subset_transform_matrix_cluster_label, boundary_radius_threshold)

                            boundary_radius_npy[cluster_No] = boundary_radius
                            cluster_label[cluster_No] = label_cluster

                test_rep_pred = kmeans_all_reps.predict(test_reps)
                print('test_rep_pred: {}'.format(test_rep_pred.shape))
                transform_test_matrix = kmeans_all_reps.transform(test_reps)

                augment_heights = []
                augment_widths = []
                augment_labels = []
                count1 = 0
                count2 = 0

                for test_rep_no in range(test_rep_pred.shape[0]):

                    cluster_No = test_rep_pred[test_rep_no]
                    distance_rep = transform_test_matrix[test_rep_no, cluster_No]

                    if boundary_radius_npy[cluster_No] > 0:
                        count1 += 1
                        if distance_rep < boundary_radius_npy[cluster_No]:
                            count2 += 1
                            augment_heights.append(test_height_starts[test_rep_no])
                            augment_widths.append(test_width_starts[test_rep_no])
                            augment_labels.append(cluster_label[cluster_No])
                print("The number of valid image patches: {}".format(count2))

                print("#################### CREATE AND SAVE MAP")
                ## CREATE MAP
                data_array_test = np.load(os.path.join(config.NUMPY_DIR, f"CDL_{year}_{ROI}_{cropName}.npy"))
                data_array_test_shape = data_array_test.shape
                map_count = np.zeros((3, data_array_test_shape[0],data_array_test_shape[1])).astype(np.uint8)

                for test_patch_no in range(len(augment_heights)):
                    height_start = augment_heights[test_patch_no]
                    width_start = augment_widths[test_patch_no]
                    map_count[augment_labels[test_patch_no], height_start:height_start+config.input_patch_size,width_start:width_start+config.input_patch_size] += 1

                with open(os.path.join(config.MAP_COUNT_DIR,
                                       f"{ROI}_{model_ROI}_{time_stops_pred}_{config.input_patch_size}_{year}{CA_suffix}_{cropName}_MAP_COUNT_{itr}"),'wb') as f: #c
                    pickle.dump(map_count, f)

                print(f"SAVE MAP_COUNT {itr}")

                del map_count
                gc.collect()