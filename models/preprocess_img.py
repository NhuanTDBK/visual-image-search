import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

def compute_mean_std():
    import glob
    import numpy as np
    from tqdm import tqdm

    mean_ = np.zeros(3, dtype=np.float)
    std_ = np.zeros(3, dtype=np.float)

    ncount = 0
    npage = 0

    img_size = (380, 380)
    batch_size = 1024

    img_lst = np.zeros((batch_size, *img_size, 3))
    for fname in tqdm(glob.glob("train_images/*")):
        img = tf.image.decode_jpeg(open(fname, "rb").read())
        img = tf.image.resize(img, img_size)
        img_lst[ncount] = img.numpy()
        npage += 1
        ncount += 1

        if npage == batch_size:
            mean_ += np.mean(img_lst, axis=(0, 1, 2))
            std_ += np.std(img_lst, axis=(0, 1, 2))
            ncount = 0

    return mean_ / ncount, std_ / ncount
