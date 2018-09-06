import sys
import pickle


def convert_time(str_time):
    m, s = map(int, str_time.split(':'))
    return m * 60 + s


def seconds_to_time(sec):
    return '{}:{:02}'.format(sec // 60, sec % 60)


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


def start_table_to_dict(table):
    result = {}
    for _, v in table.iterrows():
        name = v.file_name
        result[name] = v.first_start, v.second_start
    return result


def cur_time(i_frame, start, FRAME_RATE=1):
    time = i_frame * FRAME_RATE
    if time > start[1]:
        return time - start[1] + 45 * 60
    return time - start[0]


def cur_frame(time, start):
    if time > 45 * 60:
        return time - 45 * 60 + start[1]
    return time + start[0]