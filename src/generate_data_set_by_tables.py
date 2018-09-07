import pandas as pd
from lib.utils import parse_args, convert_time, start_table_to_dict, end_table_to_dict, cur_time, seconds_to_time

EVENTS_TYPE = ['удар по воротам', 'угловой', 'замена', 'желтая карточка', 'гол']

if __name__ == '__main__':
    args = parse_args()

    train_table_path = args['train_table']
    time_table_path = args['time_table']
    output = args['output']

    time_table = pd.read_csv(time_table_path)
    starts_dict = start_table_to_dict(time_table)
    ends_dict = end_table_to_dict(time_table)

    train_table = pd.read_csv(train_table_path)

    data_set = []
    for _, x in train_table.iterrows():
        if x.event_type not in EVENTS_TYPE:
            continue
        starts = starts_dict['resized_' + x.file_name]
        ends = starts_dict['resized_' + x.file_name]
        event_time_video = convert_time(x.event_time)
        event_time = cur_time(event_time_video, starts, ends)
        data_set.append({
            'file_name': x.file_name,
            'event_type': x.event_type,
            'event_time': seconds_to_time(event_time)
        })
    pd.DataFrame(data_set).to_csv(output)
