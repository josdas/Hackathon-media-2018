from lib.utils import parse_args, convert_time, cur_time, cur_frame, start_table_to_dict
from lib.video import get_reader, get_total_len
from lib.utils import save_pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

FRAME_RATE = 1


def gen_y(y_table, n_frames, start, win=30):
    y = [None] * n_frames
    dists = np.ones(n_frames) * np.inf
    for _, v in y_table.iterrows():
        type = v.event_type
        time = v.event_time
        i_frame = cur_frame(time, start)
        for i in range(max(i_frame - win, 0), min(i_frame + win, n_frames)):
            dist = np.abs(i - i_frame)
            if dists[i] > dist:
                dists[i] = dist
                y[i] = type
    return y


if __name__ == '__main__':
    args = parse_args()
    files = args['files']

    start_table_path = args['start_table']
    start_table = start_table_to_dict(pd.read_csv(start_table_path))

    y_train_path = args['y_train']
    y_train = pd.read_csv(y_train_path)
    y_train['event_time'] = y_train['event_time'].apply(convert_time)

    dir = args.get('dir', '.')
    if dir[-1] != '/':
        dir = dir + '/'

    output_path = args.get('output', '.')
    if output_path[-1] != '/':
        output_path = output_path + '/'

    for file in files:
        if file + '.pickle' in os.listdir(output_path):
            print('Skip', file)
            continue
        X, X_time = [], []
        start = start_table['resized_' + file]
        path = dir + 'resized_' + file
        y_table = y_train[y_train['file_name'] == file]
        y = gen_y(y_train, get_total_len(path), start)
        for i, frame in tqdm(enumerate(get_reader(path))):
            time = cur_time(i, start)
            X.append(frame)
            X_time.append(time)
        print(len(X_time), X_time[0], X_time[-1], start)
        save_pickle((X, y, X_time), output_path + file + '.pickle')
