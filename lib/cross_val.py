from lib.scoring import task_score
import numpy as np
from tqdm import tqdm_notebook as tqdm


def cross_val_score(model, data, files, cv):
    files = np.array(files)
    scores = []
    for train, test in tqdm(cv.split(files)):
        model.fit(files[train])
        y_pred = model.predict(files[test])
        y_true = data[data['file_name'].isin(files[test])]
        score = task_score(y_true, y_pred)
        scores.append(score)
        print('cross_val_score: score={} test_files={}'.format(score, files[test]))
    return np.array(scores)
