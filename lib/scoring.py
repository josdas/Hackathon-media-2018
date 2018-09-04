import pandas as pd


def find_match(row, y_true):
    right_time = row.event_time
    similiar_events = y_true[(y_true.file_name == row.file_name) & (y_true.event_type == row.event_type)]
    for index, event in similiar_events.iterrows():
        if event.checked:
            continue
        else:
            if abs(right_time - event.event_time) < 60:
                y_true.at[index, 'checked'] = True
                return True
    return False


def task_score(y_true, y_pred):
    """
    :param y_true: pandas.DataFrame with columns: 'event_time', 'event_type', 'file_name'. Time in seconds.
    :param y_pred: like y_true
    :return: f1 score after matching
    """
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    y_true['checked'] = False
    y_pred['result'] = y_pred.apply(lambda row: find_match(row, y_true), axis=1)
    true_positives = len(y_pred[y_pred.result])
    false_positives = len(y_pred[~y_pred.result])
    false_negatives = len(y_true) - true_positives
    return true_positives / (true_positives + false_positives + false_negatives)


if __name__ == '__main__':
    # Test 1
    y_true = pd.DataFrame()
    y_true['event_time'] = [0, 100, 110]
    y_true['event_type'] = [1, 1, 1]
    y_true['file_name'] = [1, 1, 1]

    y_pred = pd.DataFrame()
    y_pred['event_time'] = [0, 165, 1000, 3000]
    y_pred['event_type'] = [1, 1, 1, 1]
    y_pred['file_name'] = [1, 1, 1, 1]
    score = task_score(y_true, y_pred)
    assert(score == 2 / (2 + 2 + 1))

    # Test 2
    y_true = pd.DataFrame()
    y_true['event_time'] = [0, 100, 110]
    y_true['event_type'] = [1, 1, 1]
    y_true['file_name'] = [1, 1, 1]

    y_pred = pd.DataFrame()
    y_pred['event_time'] = [0, 165, 155 , 3000]
    y_pred['event_type'] = [1, 1, 1, 1]
    y_pred['file_name'] = [1, 1, 1, 1]
    score = task_score(y_true, y_pred)
    assert(score == 3 / (3 + 1))

    #Test 3
    y_true = pd.DataFrame()
    y_true['event_time'] = [0, 100, 110]
    y_true['event_type'] = [1, 1, 2]
    y_true['file_name'] = [1, 1, 1]

    y_pred = pd.DataFrame()
    y_pred['event_time'] = [0, 165, 155, 3000]
    y_pred['event_type'] = [1, 1, 2, 1]
    y_pred['file_name'] = [1, 1, 1, 1]
    score = task_score(y_true, y_pred)
    assert(score == 2 / (2 + 2 + 1))
