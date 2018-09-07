import pandas as pd
import numpy as np
import pickle
import os

from lib.video import *
from lib.utils import convert_time, start_table_to_dict, cur_frame, cur_time

UI_REAL = '../data/train/real_start_train.csv'
UI_PREDICTION = '../output/features/games_starts_train.csv'
VIDEO_DIR = '../data/train/'
NONE_TYPE = 'NONE'

FILES = [
    'resized_639900_5.mp4',
    'resized_639919_5.mp4',
    'resized_639933_5.mp4',
    'resized_639939_5.mp4',
    'resized_640085_5.mp4',
    'resized_640196_5.mp4',
    'resized_641579_3.mp4',
    'resized_643734_5.mp4'
]

full_event_table = pd.read_csv(UI_REAL) \
    .drop(["first_start", "first_end", "second_start", "second_end"], axis=1) \
    .dropna()
full_event_table = full_event_table[full_event_table['file_name'].isin(FILES)]

FILES = full_event_table['file_name'].unique()
','.join(FILES)

HL, WL, H, W = 90, 105, 110, 130


def gen_const_mask():
    mask = np.zeros((120, 240), 'bool')
    mask[94:101, 112:128] = True
    # mask[5:30, 20:50] = True
    # mask[10:20, 20:35] = True
    return mask


def gen_const_mask3():
    mask = np.zeros((120, 240, 3), 'bool')
    mask[94:101, 112:128] = np.ones(3, 'bool')
    # mask[5:30, 20:50] = True
    # mask[10:20, 20:35] = True
    return mask


UI_MASK = gen_const_mask()


def preproc_same(frame):
    frame = np.array(frame, dtype='float')
    return frame


def preproc_mask(frame):
    frame = np.array(frame, dtype='float')
    frame[~UI_MASK] = 0
    frame = frame[HL:H, WL:W]
    return frame


PREPROC_FUN = preproc_mask
IS_FLATTEN = True
NONK = 20
NN = 1000
UI_EVENT = 'ui'


def create_Xy(file_name):
    path = VIDEO_DIR + file_name
    n_frame = get_total_len(path)
    events_type = [None] * n_frame
    for _, row in full_event_table.iterrows():
        if row.file_name != file_name:
            continue

        def proc(l, r, event_type):
            print(event_type, (l, r), r - l + 1)
            for i in range(l, r + 1):
                events_type[i] = event_type

        l, r = convert_time(row.first_ui_start) + 3, convert_time(row.first_ui_end) - 1
        l2, r2 = convert_time(row.second_ui_start) + 3, convert_time(row.second_ui_end) - 1
        proc(l, r, UI_EVENT)
        proc(l2, r2, UI_EVENT)

    X, y = [], []
    bad_pairs = []
    shape_frame = ()
    for frame, event_type in (zip(get_reader(path), events_type)):
        frame = PREPROC_FUN(frame)
        shape_frame = frame.shape
        if IS_FLATTEN:
            frame = frame.reshape((-1))
        if event_type is None:
            bad_pairs.append((frame, NONE_TYPE))
        else:
            y.append(event_type)
            X.append(frame)
    print(shape_frame)
    n_pairs = NN
    for i in np.random.choice(range(len(bad_pairs)), n_pairs, replace=False):
        X.append(bad_pairs[i][0])
        y.append(bad_pairs[i][1])
    return X, y


def create_dataset(files):
    X, y, file_name = [], [], []
    for file in files:
        X_file, y_file = create_Xy(file)
        print('create_dataset: len(y_file)={}'.format(len(y_file)))
        X += X_file
        y += y_file
        file_name += [file] * len(y_file)
    return np.array(X), np.array(y), np.array(file_name)


X, y, file_name = create_dataset(FILES)

from sklearn.linear_model import RidgeClassifier

print('Fit')
ridge = RidgeClassifier(normalize=True, class_weight='balanced').fit(X, y)

from lib.utils import save_pickle

save_pickle(ridge, '../output/models/model_ui.pickle')
