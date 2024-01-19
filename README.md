# early-prediction_distribution-yield

config: path & parameter configuration

DATA: data downloading & preprocessing
- download_S2: download Sentinel-2 data by GEE
- INTERPOLATION: mosaic images; interpolation to fill up non values.
- PREPROCESS_DATA: data validation mask; rescale pixel values to 0-1