import sys
import config
import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import time
import pickle
import MODEL
import DATA_LOADER

SWITCH_SPLIT = True
config.batch_size = 1024*3
train_ROI_list = ["NW", "SW", "NC", "SC", "NE", "SE"]
cropName = "CS"
config.n_epochs = 400
epoch_threshold = 30
train_ROI_name = "ALL"

for train_ROI in train_ROI_list:
    print("ROI: ", train_ROI)

def mse_loss(pred_image, target, reduction):
    mask_input = torch.isnan(target)
    diff_square = torch.square(torch.sub(pred_image, target))
    prod = torch.mul(~mask_input, diff_square)
    sum_se = torch.nansum(prod, dim=[1, 2, 3, 4]).float()
    valid_sum = torch.sum(~mask_input, dim=[1, 2, 3, 4]).float()
    norm_sum = sum_se / valid_sum

    return norm_sum

def cl_criterion(reps, no_max_reps): # the criterion of constrained clustering
    no_of_reps = reps.shape[0]
    torch_arr = torch.randint(0, no_of_reps,
                              (2 * no_max_reps,))  # 2*no_max_reps: randonly select 2*no_max_reps encoder pairs
    torch_arr_2 = torch.randint(0, no_of_reps, (2 * no_max_reps,))
    total_log_sum = 0
    count_rep = 0

    for i in range(torch_arr.shape[0]):
        if (torch_arr[i] == torch_arr_2[i]):
            continue

        # Denominator
        mag1 = torch.sqrt(torch.sum(torch.square(reps[torch_arr[i]])))
        mag2 = torch.sqrt(torch.sum(torch.square(reps[torch_arr_2[i]])))
        prod12 = mag1 * mag2

        # Numerator
        dot12 = torch.abs(torch.dot(reps[torch_arr[i]], reps[torch_arr_2[i]]))

        # cos and log
        cos12 = torch.div(dot12, prod12)
        log12 = -torch.log(cos12)

        if (torch.isnan(log12)):
            print(f"log12: {log12}, cos12: {cos12}")
            print(f"mag1: {mag1},mag2: {mag2},prod12: {prod12}")
            print(f"dot12: {dot12}")
        total_log_sum += log12
        count_rep += 1

    if (count_rep == 0):
        count_rep = 1
    avg_log = torch.div(total_log_sum, count_rep)

    return avg_log

# patches_of_each_class = np.bincount(np.array(data_train.label_ids))
# print(f"QUALIFIED PATCHES FOR EACH CLASS IN DATA TRAIN: {patches_of_each_class}")

print("########## BUILD MODEL")
model = MODEL.UNET_LSTM_BIDIRECTIONAL_AUTOENCODER_TACA_4(in_channels=config.channels, out_channels=config.channels)
model = model.to('cuda')
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print("########## TRAIN MODEL")
train_loss = []
val_loss = []
val_best_corn = float('inf')
val_best_soy = float('inf')
val_best_others = float('inf')
val_best_recon_loss = float('inf')

train_best_corn = float('inf')
train_best_soy = float('inf')
train_best_others = float('inf')
train_best_recon_loss = float('inf')

for epoch in range(1, config.n_epochs + 1):

    print('## EPOCH {} ##'.format(epoch))
    model.train()

    epoch_loss = 0
    epoch_loss_recon = 0
    epoch_loss_corn_log = 0  #
    epoch_loss_soybean_log = 0
    epoch_loss_others_log = 0

    for train_ROI in train_ROI_list:
        with open(os.path.join(config.NUMPY_DIR, "dataset", "year_stop",
                               f"{train_ROI}_{config.input_patch_size}_trainDataset_{cropName}"), 'rb') as f:
            data_train = pickle.load(f)
        data_loader_train = DATA_LOADER.DataLoader_diffLen_1(dataset=data_train, batch_size=config.batch_size,
                                                             shuffle_batches=True, shuffle_within_batch=True)

        for batch, batch_tuple in enumerate(data_loader_train):
            # print('## BATCH TRAIN {} ##'.format(batch), end="\t")

            image_patch, label_batch = zip(*batch_tuple)

            image_patch = torch.tensor(np.array(image_patch))
            label_batch = torch.tensor(np.array(label_batch))
            # print(image_patch.shape, label_batch.shape)
            optimizer.zero_grad()

            label_patch = image_patch.to('cuda').float()
            image_patch[torch.isnan(image_patch)] = 0  # fill gaps

            code_vec, out_recon, _, _ = model(image_patch.to('cuda').float())
            # print("code_vec.shape: ", code_vec.shape)
            # s code_vec: (batch_size, encoder_dimension)
            # s out_recon: (batch_size, seq_len, channels, input_patch_size, input_patch_size)

            code_vec_others = code_vec[label_batch == 0]  # others encoder
            code_vec_corn = code_vec[label_batch == 1]  # corn encoder
            code_vec_soybean = code_vec[label_batch == 2]  # soybean encoder
            min_class_labels = np.min([code_vec_others.shape[0], code_vec_corn.shape[0], code_vec_soybean.shape[0]])
            # print(code_vec_corn.shape[0],code_vec_soybean.shape[0], min_class_labels) #p 26, 111, 26

            if (code_vec_corn.shape[0] > 2):
                corn_batch_loss_log = cl_criterion(code_vec_corn, min_class_labels)  # constrained clustering for corn
            else:
                corn_batch_loss_log = torch.zeros(1, requires_grad=True)

            if (code_vec_soybean.shape[0] > 2):
                soybean_batch_loss_log = cl_criterion(code_vec_soybean,
                                                      min_class_labels)  # constrained clustering for soybean
            else:
                soybean_batch_loss_log = torch.zeros(1, requires_grad=True)

            if (code_vec_others.shape[0] > 2):
                others_batch_loss_log = cl_criterion(code_vec_others,  # constrained clustering for others
                                                     min_class_labels)
            else:
                others_batch_loss_log = torch.zeros(1, requires_grad=True)

            batch_loss_recon = torch.mean((mse_loss(pred_image=out_recon, target=label_patch, reduction='None')))

            if epoch < epoch_threshold:  #
                batch_loss = batch_loss_recon
            else:
                batch_loss = batch_loss_recon + (soybean_batch_loss_log + corn_batch_loss_log+others_batch_loss_log)*0.1

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_loss_recon += batch_loss_recon.item()
            epoch_loss_corn_log += corn_batch_loss_log.item()
            epoch_loss_soybean_log += soybean_batch_loss_log.item()
            epoch_loss_others_log += others_batch_loss_log.item()

    epoch_loss = epoch_loss / (batch+1)
    epoch_loss_recon = epoch_loss_recon / (batch+1)
    epoch_loss_corn_log = epoch_loss_corn_log / (batch+1)
    epoch_loss_soybean_log = epoch_loss_soybean_log / (batch+1)
    epoch_loss_others_log = epoch_loss_others_log / (batch+1)
    print("TRAIN: ", epoch_loss, epoch_loss_recon, epoch_loss_corn_log, epoch_loss_soybean_log, epoch_loss_others_log)
    train_loss.append(epoch_loss)

    if epoch >= epoch_threshold:
        if (epoch_loss_recon <= train_best_recon_loss) & (epoch_loss_corn_log <= train_best_corn) & (epoch_loss_soybean_log <= train_best_soy) & (epoch_loss_others_log <= train_best_others):
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"best_{train_ROI_name}_{cropName}_TACA_{epoch_threshold}_{epoch}.pt"))
            train_best_corn = epoch_loss_corn_log
            train_best_soy = epoch_loss_soybean_log
            train_best_others = epoch_loss_others_log
            train_best_recon_loss = epoch_loss_recon

    # # VALIDATION
    # model.eval()
    # epoch_loss_val = 0
    # epoch_loss_recon_val = 0
    # epoch_loss_corn_log_val = 0
    # epoch_loss_soybean_log_val = 0
    # epoch_loss_others_log_val = 0
    #
    # for batch, batch_tuple in enumerate(data_loader_val):
    #     # print('## BATCH VAL {} ##'.format(batch), end="\t")
    #
    #     image_patch, label_batch = zip(*batch_tuple)
    #
    #     image_patch = torch.tensor(np.array(image_patch))
    #     label_batch = torch.tensor(np.array(label_batch))
    #
    #     label_patch = image_patch.to('cuda').float()
    #     image_patch[torch.isnan(image_patch)] = 0
    #
    #     code_vec, out_recon, _, _ = model(image_patch.to('cuda').float())
    #
    #     code_vec_others = code_vec[label_batch == 0]
    #     code_vec_corn = code_vec[label_batch == 1]
    #     code_vec_soybean = code_vec[label_batch == 2]
    #     min_class_labels = np.min([code_vec_others.shape[0], code_vec_corn.shape[0], code_vec_soybean.shape[0]])
    #
    #     if (code_vec_corn.shape[0] > 2):
    #         corn_batch_loss_log_val = cl_criterion(code_vec_corn, min_class_labels)
    #     else:
    #         corn_batch_loss_log_val = torch.zeros(1, requires_grad=True)
    #
    #     if (code_vec_soybean.shape[0] > 2):
    #         soybean_batch_loss_log_val = cl_criterion(code_vec_soybean,
    #                                                   min_class_labels)
    #     else:
    #         soybean_batch_loss_log_val = torch.zeros(1, requires_grad=True)
    #
    #     if (code_vec_others.shape[0] > 2):
    #         others_batch_loss_log_val = cl_criterion(code_vec_others, min_class_labels)
    #     else:
    #         others_batch_loss_log_val = torch.zeros(1, requires_grad=True)
    #
    #     batch_loss_recon_val = torch.mean((mse_loss(pred_image=out_recon, target=label_patch, reduction='None')))
    #
    #     if epoch < epoch_threshold:  #
    #         batch_loss_val = batch_loss_recon_val
    #     else:
    #         batch_loss_val = batch_loss_recon_val + (
    #             soybean_batch_loss_log_val + corn_batch_loss_log_val + others_batch_loss_log_val)*0.1
    #
    #     epoch_loss_val += batch_loss_val.item()
    #     epoch_loss_recon_val += batch_loss_recon_val.item()
    #     epoch_loss_corn_log_val += corn_batch_loss_log_val.item()
    #     epoch_loss_soybean_log_val += soybean_batch_loss_log_val.item()
    #     epoch_loss_others_log_val += others_batch_loss_log_val.item()
    #
    # epoch_loss_val = epoch_loss_val / (batch)
    # epoch_loss_recon_val = epoch_loss_recon_val / (batch)
    # epoch_loss_corn_log_val = epoch_loss_corn_log_val / (batch)
    # epoch_loss_soybean_log_val = epoch_loss_soybean_log_val / (batch)
    # epoch_loss_others_log_val = epoch_loss_others_log_val / (batch)
    # print("VAL: ", epoch_loss_val, epoch_loss_recon_val, epoch_loss_corn_log_val, epoch_loss_soybean_log_val,
    #       epoch_loss_others_log_val)
    # val_loss.append(epoch_loss_val)
    #
    # # if epoch >= 100:
    #     # torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"ULBA_TACA_{config.input_patch_size}", f"{train_ROI}_{epoch}_{cropName}_TACA.pt"))
    # if epoch >= epoch_threshold:
    #     if (epoch_loss_corn_log_val <= val_best_corn) & (epoch_loss_soybean_log_val <= val_best_soy) & (epoch_loss_recon_val <= val_best_recon_loss) & (epoch_loss_others_log_val <= val_best_others):
    # #     if (epoch_loss_corn_log_val < val_best_corn) & (epoch_loss_soybean_log_val < val_best_soy):
    #         torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"ULBA_TACA_{config.input_patch_size}", f"best_{val_ROI}_{train_ROI}_{cropName}_TACA_{epoch}.pt"))
    #         val_best_corn = epoch_loss_corn_log_val
    #         val_best_soy = epoch_loss_soybean_log_val
    #         val_best_others = epoch_loss_others_log_val
    #         val_best_recon_loss = epoch_loss_recon_val