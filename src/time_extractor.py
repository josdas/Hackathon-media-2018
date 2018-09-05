import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from lib.video import get_reader, get_total_len, get_video_shape
from lib.utils import parse_args

GREEN_THRESHOLD = 0.4
UI_PROB_THRESHOLD = 0.85
N_TOP_COLORS = 50

FRAME_RATE = 4
SKIP_BEGGING = 0
MAX_FRAMES_COUNT = None

dx = (0, 0, 1, -1, -1, 1, 1, -1)
dy = (1, -1, 0, 0, -1, 1, -1, 1)
GOOD_COLORS = (0, 1, 2)


def create_video_iter(path):
    return tqdm(get_reader(path, FRAME_RATE, skip=SKIP_BEGGING, size=MAX_FRAMES_COUNT),
                total=get_total_len(path, FRAME_RATE, skip=SKIP_BEGGING, size=MAX_FRAMES_COUNT))


def calc_green_pixels(frame):
    frame = frame.astype('float')
    return frame[:, :, 1] * 2 > frame[:, :, 0] + frame[:, :, 2]


def calc_green_counts(path):
    total = 0
    counts = np.zeros(get_video_shape(path))
    for frame in create_video_iter(path):
        counts += calc_green_pixels(frame)
        total += 1
    return counts / total


def extract_green_mask(green_counts, threshold):
    return green_counts < threshold


def clean_borders(mask, size=4):
    mask = mask.copy()
    borders = np.ones_like(mask)
    borders[size:-size, size:-size] = 0
    mask[borders] = 0
    return mask


def color_discretization(color, k=5):
    return tuple(color // k * k)


def create_colors_dict(path, green_mask):
    colors_dict = defaultdict(Counter)
    total = 0
    for frame in create_video_iter(path):
        for i in range(green_mask.shape[0]):
            for j in range(green_mask.shape[1]):
                if green_mask[i, j]:
                    colors_dict[i, j][color_discretization(frame[i, j])] += 1
        total += 1
    return colors_dict


def generate_static_pixels_mask(colors_dict, green_mask):
    mask = np.zeros_like(green_mask)
    for pixel, counter in colors_dict.items():
        mc = counter.most_common(N_TOP_COLORS)
        total = sum(v for k, v in counter.items())
        prob = sum(v for k, v in mc) / total
        mask[pixel] = prob > 0.5
    return mask


def create_top_color_dict(colors_dict, mask):
    return {
        pixel: {k for k, v in counter.most_common(N_TOP_COLORS)}
        for pixel, counter in colors_dict.items()
        if mask[pixel]
    }


def ui_probability(frame, colors_dict, mask):
    sim_mask = np.zeros_like(mask, dtype='uint8')
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                if color_discretization(frame[i, j]) in colors_dict[i, j]:
                    for k in range(len(dx)):
                        p = i + dx[k], j + dy[k]
                        if mask[p]:
                            sim_mask[p] = 1
    prob = sim_mask.sum() / mask.sum()
    return prob


def find_two_segments(path, colors_dict, mask):
    probs = [ui_probability(frame, colors_dict, mask)
             for frame in create_video_iter(path)]
    l = 45 * 60 // FRAME_RATE  # 45min in frames
    n = len(probs)
    pref = np.cumsum(np.array(probs) > UI_PROB_THRESHOLD)
    mx = -1
    best_l1, best_l2 = 0, 0
    for l1 in range(n - l):
        for l2 in range(l1 + l, n - l):
            s1, s2 = pref[l1 + l] - pref[l1], pref[l2 + l] - pref[l2]
            if mx < s1 + s2:
                mx = s1 + s2
                best_l1, best_l2 = l1, l2
    return best_l1 * FRAME_RATE, best_l2 * FRAME_RATE


def find_game_starts(path):
    green_counts = calc_green_counts(path)
    green_mask = extract_green_mask(green_counts, GREEN_THRESHOLD)
    green_mask = clean_borders(green_mask)
    colors_dict = create_colors_dict(path, green_mask)
    static_mask = generate_static_pixels_mask(colors_dict, green_mask)
    colors_dict = create_top_color_dict(colors_dict, static_mask)
    return find_two_segments(path, colors_dict, static_mask)


if __name__ == '__main__':
    args = parse_args()
    files = args['files']
    dir = args.get('dir', '.')
    if dir[-1] != '/':
        dir = dir + '/'
    output_path = args.get('output', 'game_start.csv')
    results = []
    for file in tqdm(files):
        path = dir + file
        print(path)
        start = find_game_starts(path)
        results.append({
            'file_name': file,
            'first_start': start[0],
            'second_start': start[1]
        })
        pd.DataFrame(results).to_csv(output_path)
