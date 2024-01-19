import os
year_list = [2019, 2020, 2021, 2022, 2023]
year_test = 2022
year_pred = 2023

# FILES INFO
DATA_DIR = "/scratch.global/yin00406/early-prediction/DATA/interpolation"
RESULT_DIR = "/scratch.global/yin00406/early-prediction/RESULT"
MODEL_DIR = "/home/jinzn/yin00406/early-prediction/method_embedding/MODEL"
CDL_DIR = "/home/jinzn/yin00406/early-prediction/method_embedding/DATA/CDL"
NUMPY_DIR = os.path.join(RESULT_DIR, "NUMPY")
PRED_MAP = os.path.join(RESULT_DIR, "PRED_MAP")
MAP_COUNT_DIR = os.path.join(RESULT_DIR, "MAP_COUNT")
CLS_PRED_MAP = os.path.join(RESULT_DIR, "CLS_PRED_MAP")

grids_NW = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
grids_SW = ["D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3"]
grids_NC = ["A4", "B4", "B5", "B6", "C4", "C5", "C6"]
grids_SC = ["D4", "D5", "D6", "E4", "E5", "E6", "F4", "F5", "F6"]
grids_NE = ["C7", "C8", "C9", "D7", "D8", "D9", "D10"]
grids_SE = ["E7", "E8", "E9", "E10", "F7", "F8", "F9"]

classes = ["others", "corn", "soybean"]
step = 0
# steps = range(0,24)
doys_CS = range(150, 266,5)
doys_CS_leap = range(151, 267, 5)
doys_CS_pred = range(150, 236, 5)

extended_doys_CS = range(120, 266, 5)
extended_doys_CS_leap = range(121, 267, 5)

# CHANNELS INFO
clip = 1
channel_names = ["red1", "red2", "red3", "nir", "red4", "swir1", "swir2", "REP", "RENDVI2"]
channels = len(channel_names)