import pandas as pd
from lib.utils import parse_args, seconds_to_time

TOTAL_TIME = 6000


def gen_prediction_every_n_const(files, n, event):
    y_pred = []
    for file in files:
        for time in range(n // 2, TOTAL_TIME, n):
            y = {
                'file_name': file,
                'event_type': event,
                'event_time': seconds_to_time(time)
            }
            y_pred.append(y)
    return pd.DataFrame(y_pred)


if __name__ == '__main__':
    args = parse_args()
    output = args['output']
    predict_table_path = args['predict_table']
    predict_table = pd.read_csv(predict_table_path)
    files = predict_table['file_name'].unique()
    e_1 = gen_prediction_every_n_const(files, 120, 'удар по воротам')
    e_2 = gen_prediction_every_n_const(files, 120, 'угловой')
    concat = pd.concat([predict_table, e_1, e_2], ignore_index=True)
    concat = concat[['file_name', 'event_type', 'event_time']]
    concat.to_csv(output)
