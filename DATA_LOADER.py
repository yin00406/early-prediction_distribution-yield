import sys
import os
import config
import copy
import numpy as np
import pickle
import random
from collections import Counter
import torch
from torch.utils.data.dataset import Dataset
from osgeo import gdal, gdalconst, osr
import time

def create_patches_train_random(array, valid_array, label_array, crop_labels, thresholds, current_time_step, patch_purity_threshold, input_patch_size):
    array_height, array_width = (array.shape[2], array.shape[3])
    height_start_max = array_height - input_patch_size
    width_start_max = array_width - input_patch_size

    image_patches_total = []
    label_ids_total = []

    random.seed(66)
    
    for crop_label in crop_labels:
        print("crop_label: {}".format(crop_label))
        label_count = 0
        height, width = np.where(label_array==crop_label)

        print(len(height))

        image_patches = []
        label_ids = []

        while label_count < thresholds[crop_label]:

            indices = np.random.choice(len(height), 1000, replace=False)

            for index in indices:

                height_start = height[index]
                width_start = width[index]
                if (height_start > height_start_max) or (width_start > width_start_max):
                    continue

                i_image_start, i_image_end = height_start, height_start + input_patch_size
                j_image_start, j_image_end = width_start, width_start + input_patch_size

                image_patch = array[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label_array[i_image_start:i_image_end, j_image_start:j_image_end]
                image_valid_patch = valid_array[:, i_image_start:i_image_end, j_image_start:j_image_end]

                if np.sum(image_valid_patch) >= config.valid_threshold*current_time_step*input_patch_size*input_patch_size:
                    count_array = np.bincount(label_patch.flatten())

                    if(count_array[crop_label] >= patch_purity_threshold*input_patch_size * input_patch_size):
                        image_patches.append(image_patch)
                        label_ids.append(crop_label)
                        label_count += 1

        image_patches_total += image_patches[:thresholds[crop_label]]
        label_ids_total += label_ids[:thresholds[crop_label]]

    return image_patches_total, label_ids_total

def create_patches_test(array, valid_array, patch_count, current_time_step):
    height, width = (array.shape[2], array.shape[3])
    height_start_max = height - config.input_patch_size
    width_start_max = width - config.input_patch_size
    image_patches = []
    image_height_starts = []
    image_width_starts = []
    patches_created_count = 0

    while patches_created_count < patch_count:
        # print(patches_created_count)
        height_start = random.randint(0, height_start_max)
        width_start = random.randint(0, width_start_max)
        i_image_start, i_image_end = height_start, height_start + config.input_patch_size
        j_image_start, j_image_end = width_start, width_start + config.input_patch_size

        image_patch = array[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
        image_valid_patch = valid_array[:, i_image_start:i_image_end, j_image_start:j_image_end]

        if np.sum(image_valid_patch) >= config.valid_threshold * current_time_step * config.input_patch_size * config.input_patch_size:
            image_patches.append(image_patch)
            image_height_starts.append(i_image_start)
            image_width_starts.append(j_image_start)
            patches_created_count += 1

    # print(f"The number of clipped patches: {patches_created_count}")
    return image_patches, image_height_starts, image_width_starts

class SEGMENTATION_train(Dataset):

    def __init__(self, image_patches, label_ids):
        self.image_patches = image_patches
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, index):
        return self.image_patches[index], self.label_ids[index]

class SEGMENTATION_test(Dataset):

    def __init__(self, image_patches, image_height_starts, image_width_starts):
        self.image_patches = image_patches
        self.image_height_starts = image_height_starts
        self.image_width_starts = image_width_starts

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, index):
        return self.image_patches[index], self.image_height_starts[index], self.image_width_starts[index]

class DataLoader_diffLen:
    def __init__(self, dataset, batch_size, shuffle_within_batch=True, shuffle_batches=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_within_batch = shuffle_within_batch
        self.shuffle_batches = shuffle_batches
        self.batches = self.create_batches()

    def create_batches(self):
        # Create batches
        num_batches = len(self.dataset) // self.batch_size
        scale = int(1024*10*3/self.batch_size)
        batches_large = [list(range(i * self.batch_size * scale, (i + 1) * self.batch_size * scale)) for i in range(int(num_batches/scale))]

        # Shuffle within each batch
        if self.shuffle_within_batch:
            for i, batch_large in enumerate(batches_large):
                random.shuffle(batch_large)
                batches_large[i] = batch_large[:self.batch_size]

        # Shuffle the order of batches
        if self.shuffle_batches:
            random.seed(66)
            random.shuffle(batches_large) #c

        return batches_large

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= len(self.batches):
            raise StopIteration

        # Get next batch
        batch_indexes = self.batches[self.batch_idx]
        self.batch_idx += 1

        # Fetch data from dataset
        data = [self.dataset[i] for i in batch_indexes]
        return data

    def __len__(self):
        return len(self.batches)

class DataLoader_diffLen_1:
    def __init__(self, dataset, batch_size, shuffle_within_batch=True, shuffle_batches=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_within_batch = shuffle_within_batch
        self.shuffle_batches = shuffle_batches
        self.batches = self.create_batches()

    def create_batches(self):
        # Create batches
        num_batches = len(self.dataset) // self.batch_size
        batches = [list(range(i * self.batch_size, (i + 1) * self.batch_size)) for i in range(num_batches)]

        # Shuffle within each batch
        if self.shuffle_within_batch:
            # random.seed(66)
            for i, batch in enumerate(batches):
                random.shuffle(batch)
        
        # Shuffle the order of batches
        if self.shuffle_batches:
            random.seed(66)
            random.shuffle(batches) #c

        return batches

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= len(self.batches):
            raise StopIteration

        batch_indexes = self.batches[self.batch_idx]
        self.batch_idx += 1

        data = [self.dataset[i] for i in batch_indexes]
        return data

    def __len__(self):
        return len(self.batches)
