# early-prediction_distribution-yield

config: path & parameter configuration

DATA: data downloading & preprocessing
- download_S2: download Sentinel-2 data by GEE
- INTERPOLATION: mosaic images; interpolation to fill up non values.
- PREPROCESS_DATA: data validation mask; rescale pixel values to 0-1
- GENERATE_QUALIFIED_DATASET: prepare data for training and test

MODEL

- MODEL: model structure
- TRAIN: model training

PREDICT

- SAMPLE_PREDICTION: predict classes for image patches
- RECON_HQ_SAMPLE_MAP: reconstruct maps for predicted samples