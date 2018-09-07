import sys
import pickle


def convert_time(str_time):  # TODO CHECK
    time = list(map(int, str_time.split(':')))
    if len(time) > 2:
        h, m, s = time
        return h * 60 * 60 + m * 60 + s
    m, s = time
    return m * 60 + s


def seconds_to_time(sec):  # TODO CHECK
    return '{:02}:{:02}'.format(sec // 60, sec % 60)


def seconds_to_hours(sec):  # TODO CHECK
    if sec >= 60 * 60:
        h = sec // (60 * 60)
        sec -= h * 60 * 60
        return str(h) + ':' + seconds_to_time(sec)
    return seconds_to_time(sec)


def _parse_value(value):
    if len(value) == 0:
        return None
    if value[0] == '[' and value[-1] == ']':
        return value[1:-1].split(',')
    return value


def parse_args():
    args = sys.argv[1:]
    return {k: _parse_value(v) for k, v in map(lambda kv: kv.split('='), args)}


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def start_table_to_dict(table):
    result = {}
    for _, v in table.iterrows():
        name = v.file_name
        result[name] = convert_time(v.first_start), convert_time(v.second_start)
    return result


def end_table_to_dict(table):
    result = {}
    for _, v in table.iterrows():
        name = v.file_name
        result[name] = convert_time(v.first_end), convert_time(v.second_end)
    return result


ONE_TIME_SEC = 45 * 60


def cur_time(i_frame, starts, ends, FRAME_RATE=1):  # TODO CHECK
    time = i_frame * FRAME_RATE
    if time > starts[1]:
        return time - starts[1] + ONE_TIME_SEC
    return time - starts[0]


def cur_frame(time, start, ends):
    if time > ends[0] - start[0]:
        return time - (ends[0] - start[0]) + start[1]
    return time + start[0]
