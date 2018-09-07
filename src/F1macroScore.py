# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import sys
import pandas as pd
import os.path
from functools import partial

if len(sys.argv) < 3:
    print("Формат вызова: python " + sys.argv[0] + " файл-с-правильными-ответами файл-с-решением")
    sys.exit(1)

labels_file_name = sys.argv[1]
solution_file_name = sys.argv[2]
if len(sys.argv) > 3:
    remove_files = sys.argv[3] == 'true'
else:
    remove_files = False

if not os.path.isfile(labels_file_name):
    print("Не могу найти файл " + labels_file_name)
    sys.exit(1)

if not os.path.isfile(solution_file_name):
    print("Не могу найти файл " + solution_file_name)
    sys.exit(1)

try:
    labels = pd.read_csv(labels_file_name)
except:
    print("Не могу прочитать файл " + labels_file_name)
    sys.exit(1)

try:
    data = pd.read_csv(solution_file_name)
except:
    print("Не могу прочитать файл " + solution_file_name)
    sys.exit(1)

if remove_files is not None:
    labels = labels[labels['file_name'].isin(data['file_name'].unique())]


def check_file_columns(df, filename):
    if 'file_name' not in df.columns:
        print("Файл " + filename + " не содержит обязательный столбец file_name")
        sys.exit(1)
    if 'event_type' not in df.columns:
        print("Файл " + filename + " не содержит обязательный столбец event_type")
        sys.exit(1)
    if 'event_time' not in df.columns:
        print("Файл " + filename + " не содержит обязательный столбец event_time")
        sys.exit(1)


def convert_time(str_time):
    if not isinstance(str_time, str):
        return -100
    try:
        m_str, s_str = str_time.split(':')
    except ValueError:
        return -100
    try:
        m = int(m_str)
    except ValueError:
        m = -100
    try:
        s = int(s_str)
    except ValueError:
        s = -100
    return m * 60 + s


check_file_columns(labels, labels_file_name)
check_file_columns(data, solution_file_name)
f1_scores = []
EVENTS_TYPE = ['удар по воротам', 'угловой', 'замена', 'желтая карточка', 'гол']
for event_type in EVENTS_TYPE:
    edata = data[data.event_type == event_type].copy().reset_index()
    elabels = labels[labels.event_type == event_type].copy().reset_index()
    elabels['checked'] = False

    if not len(edata):
        print(event_type, 0, 0, len(elabels))
        f1_scores.append(0)
        continue


    def find_match(row, event_type):
        right_time = convert_time(row.event_time)
        similiar_events = elabels[(elabels.file_name == row.file_name)]
        for index, event in similiar_events.iterrows():
            if event.checked:
                continue
            else:
                if abs(right_time - convert_time(event.event_time)) < 60:
                    elabels.at[index, 'checked'] = True
                    return True
        return False


    edata['result'] = edata.apply(partial(find_match, event_type=event_type), axis=1)
    true_positives = len(edata[edata.result])
    false_positives = len(edata[~edata.result])
    false_negatives = len(elabels) - true_positives
    f1_score = true_positives / (true_positives + false_positives + false_negatives)
    print(event_type, true_positives, false_positives, false_negatives, f1_score)
    f1_scores.append(f1_score)

print(round(sum(f1_scores) / 5, 6))
