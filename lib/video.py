from matplotlib import pyplot as plt
import imageio


def get_total_len(path, frame_rate=1, size=None, skip=0):
    n = imageio.get_reader(path).get_meta_data()['nframes']
    result = int((1 - skip) * n / frame_rate)
    if size is not None:
        result = min(result, size)
    return result


def get_reader(path, frame_rate=1, size=None, skip=0):
    total = 0
    n = get_total_len(path)
    for i, frame in enumerate(imageio.get_reader(path).iter_data()):
        if skip is not None and i < skip * n:
            continue
        if i % frame_rate == 0:
            total += 1
            yield frame
        if size is not None and total == size:
            break


def get_video_shape(path):
    frame = next(get_reader(path))
    return frame.shape[:2]


def show_img(img, k=1):
    fig, ax = plt.subplots(figsize=(int(18 * k), int(20 * k)))
    ax.imshow(img, interpolation='nearest')
