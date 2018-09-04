from lib.utils import parse_args, seconds_to_time
import pandas as pd

EVENTS_TYPE = ['удар по воротам', 'угловой', 'замена', 'желтая карточка', 'гол']
EVENTS_COUNTS = [0.502, 0.223, 0.137, 0.09, 0.048]
TOTAL_TIME = 6000


def gen_prediction_every_n_best(files, n):
    y_pred = []
    for file in files:
        for time in range(n // 2, TOTAL_TIME, n):
            y = {
                'file_name': file,
                'event_type': 'удар по воротам',
                'event_time': seconds_to_time(time)
            }
            y_pred.append(y)
    return pd.DataFrame(y_pred)


if __name__ == '__main__':
    args = parse_args()
    result = gen_prediction_every_n_best(args['files'], 120)
    result.to_csv(args['result'])
