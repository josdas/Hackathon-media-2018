import numpy as np
import pandas as pd
from lib.video import get_reader, get_total_len
from lib.utils import convert_time, start_table_to_dict, cur_time, parse_args, save_pickle, load_pickle, \
    seconds_to_time, seconds_to_hours, end_table_to_dict
from sklearn.linear_model import RidgeClassifier

if __name__ == 'src.time_extractor':
    from tqdm import tqdm_notebook as tqdm

    print('Use tqdm_notebook')
else:
    from tqdm import tqdm


def gen_const_mask():
    mask = np.zeros(VIDEO_SHAPE, 'bool')
    mask[5:30, 20:50] = True
    return mask


def preproc_mask(frame):
    frame = np.array(frame, dtype='float')
    frame[~UI_MASK] = 0
    frame = frame[HL:H, WL:W]
    return frame


COMBO_LENGTH_FOR_EVENT = {
    'желтая карточка': 3,
    'гол': 2,
    'замена': 5
}
VIDEO_SHAPE = (120, 240)
GOOD_EVENTS = ['желтая карточка', 'гол', 'замена']
NONE_TYPE = 'NONE'
UI_MASK = gen_const_mask()
HL, WL, H, W = 5, 10, 30, 60
PREPROC_FUN = preproc_mask
IS_FLATTEN = True
NONK = 20
NN = 1000
RAND = 1


def create_X(file_name):
    path = VIDEO_DIR + 'resized_' + file_name
    X = []
    for frame in tqdm(get_reader(path)):
        frame = PREPROC_FUN(frame)
        if IS_FLATTEN:
            frame = frame.reshape((-1))
        X.append(frame)
    return X


def create_Xy(file_name, full_event_table):  # TODO CHECK
    path = VIDEO_DIR + 'resized_' + file_name
    n_frame = get_total_len(path)
    events_type = [None] * n_frame
    for _, row in full_event_table.iterrows():
        if row.file_name != file_name:
            continue
        l, r = convert_time(row.event_time) + 2, convert_time(row.event_end) - 1
        for i in range(l, r + 1):
            events_type[i] = row.event_type
    X, y = [], []
    bad_pairs = []
    for frame, event_type in tqdm(zip(get_reader(path), events_type)):
        frame = PREPROC_FUN(frame)
        if IS_FLATTEN:
            frame = frame.reshape((-1))
        if event_type is None:
            bad_pairs.append((frame, NONE_TYPE))
        else:
            y.append(event_type)
            X.append(frame)
    # n_pairs = min(int(len(y) * NONK), len(bad_pairs))
    n_pairs = NN
    for i in np.random.choice(range(len(bad_pairs)), n_pairs, replace=False):
        X.append(bad_pairs[i][0])
        y.append(bad_pairs[i][1])
    return X, y


def create_dataset_X(files):  # TODO CHECK
    X, file_name = [], []
    for file in tqdm(files):
        X_file = create_X(file)
        print(f'create_dataset_X: len(X_file)={len(X_file)}')
        X += X_file
        file_name += [file] * len(X_file)
    return np.array(X), np.array(file_name)


def create_dataset_Xy(files, full_event_table):  # TODO CHECK
    X, y, file_name = [], [], []
    for file in tqdm(files):
        X_file, y_file = create_Xy(file, full_event_table)
        print(f'create_dataset_Xy: len(y_file)={len(y_file)}')
        X += X_file
        y += y_file
        file_name += [file] * len(y_file)
    return np.array(X), np.array(y), np.array(file_name)


def train(files, full_event_table):
    X, y, file_name = create_dataset_Xy(files, full_event_table)
    model = RidgeClassifier(normalize=True, class_weight='balanced', random_state=RAND)
    return model.fit(X, y)


def correct_event_time(game_time, i_frame, starts):
    return seconds_to_time(game_time)


def predict(files, model, starts_dict, ends_dict):  # TODO CHECK
    X, file_name = create_dataset_X(files)
    y_pred = model.predict(X)
    combo_len = 0
    combo_type = None
    results = []
    cur_file = None
    cur_i = 0
    for file, event_type in zip(file_name, y_pred):
        if file != cur_file:
            cur_i = 0
            cur_file = file
        else:
            cur_i += 1
        if event_type == NONE_TYPE:
            combo_len = 0
            combo_type = None
            continue
        starts = starts_dict['resized_' + file]
        ends = ends_dict['resized_' + file]
        game_time = cur_time(cur_i, starts, ends)
        if cur_i < starts[0] or cur_i > ends[1] or ends[0] < cur_i < starts[1]:
            continue
        event_time = correct_event_time(game_time, cur_i, starts)
        if combo_type == event_type:
            combo_len += 1
        else:
            combo_len = 1
            combo_type = event_type
        if combo_len == COMBO_LENGTH_FOR_EVENT[combo_type]:
            results.append({
                'file_name': file,
                'event_time': event_time,
                'event_type': event_type,
                'video_time': seconds_to_hours(cur_i)
            })
            print(results[-1])
    return pd.DataFrame(results)


if __name__ == '__main__':
    args = parse_args()

    files = args['files']
    assert(len(files) == len(set(files)))
    mode = args['mode']
    assert (mode in ['train', 'test'])

    VIDEO_DIR = args.get('dir', '.')
    if VIDEO_DIR[-1] != '/':
        VIDEO_DIR = VIDEO_DIR + '/'

    output = args['output']

    if mode == 'train':
        print('Train')
        train_table_path = args['train_table']

        full_event_table = pd.read_csv(train_table_path)
        full_event_table = full_event_table[full_event_table['event_type'].isin(GOOD_EVENTS)]
        model = train(files, full_event_table)
        save_pickle(model, output)
    else:
        print('Test')
        time_table_path = args['time_table']

        model_path = args['model']

        model = load_pickle(model_path)
        time_table = pd.read_csv(time_table_path)
        starts_dict = start_table_to_dict(time_table)
        ends_dict = end_table_to_dict(time_table)
        for file in files:
            assert ('resized_' + file in starts_dict)

        prediction = predict(files, model, starts_dict, ends_dict)
        prediction.to_csv(output)
