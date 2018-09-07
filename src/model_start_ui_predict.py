import pandas as pd
import numpy as np
import pickle
import os

from lib.video import *
from lib.utils import convert_time, start_table_to_dict, cur_frame, cur_time

VIDEO_DIR = '../data/test/'
NONE_TYPE = 'NONE'

HL, WL, H, W = 90, 105, 110, 130


def gen_const_mask():
    mask = np.zeros((120, 240), 'bool')
    mask[94:101, 112:128] = True
    # mask[5:30, 20:50] = True
    # mask[10:20, 20:35] = True
    return mask


UI_MASK = gen_const_mask()


def preproc_mask(frame):
    frame = np.array(frame, dtype='float')
    frame[~UI_MASK] = 0
    frame = frame[HL:H, WL:W]
    return frame


PREPROC_FUN = preproc_mask
IS_FLATTEN = True


def create_X(file_name):
    path = VIDEO_DIR + 'resized_' + file_name
    X = []
    for frame in get_reader(path):
        frame = PREPROC_FUN(frame)
        if IS_FLATTEN:
            frame = frame.reshape((-1))
        X.append(frame)
    return X


from lib.utils import parse_args, load_pickle, seconds_to_hours

model = load_pickle('../output/models/model_ui.pickle')

results = []
output = '../data/test/real_start_test.csv'
FILES = [
    '645001_5.mp4', '645066_5.mp4', '645098_5.mp4', '645195_5.mp4', '645286_5.mp4', '645310_5.mp4', '646186_5.mp4',
    '648559_5.mp4'
]
for file in FILES:
    print(file)
    prediction = model.predict(create_X(file))
    print('predct')
    first, last = None, None
    second = None
    for i, y in enumerate(prediction):
        time = (i)
        if y == 'ui':
            if first is None:
                first = time
            last = time
    for i, y in enumerate(prediction):
        time = (i)
        if y == 'ui':
            if i < last - 44 * 60:
                second = time

    results.append({
        'file_name': 'resized_' + file,
        'first_start': seconds_to_hours(first),
        'second_start': seconds_to_hours(second)
    })
pd.DataFrame(results).to_csv(output)
