import sys


def convert_time(str_time):
    m, s = map(int, str_time.split(':'))
    return m * 60 + s


def seconds_to_time(sec):
    return '{}:{}'.format(sec // 60, sec % 60)


def _parse_value(value):
    if len(value) == 0:
        return None
    if value[0] == '[' and value[-1] == ']':
        return value[1:-1].split(',')
    return value


def parse_args():
    args = sys.argv[1:]
    return {k: _parse_value(v) for k, v in map(lambda kv: kv.split('='), args)}
