from lib.scoring import task_score
import numpy as np


def cross_val_score(model, data, files, cv):
    files = np.array(files)
    scores = []
    for train, test in cv(files):
        model.fit(files[train])
        y_pred = model.predict(files[test])
        y_true = data[data['file_name'].isin(files[test])]
        score = task_score(y_true, y_pred)
        scores.append(score)
    return np.array(scores)

