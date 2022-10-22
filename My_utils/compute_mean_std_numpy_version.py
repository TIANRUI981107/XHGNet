# -*- coding: UTF-8 -*-

"""
Compute mean and std of every channel in the whole datasets.
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def get_img_path_ls(root: str):
    """recursively glob JPG images."""
    ls = [str(p.resolve()) for p in Path(root).rglob("*.jpg")]
    return ls


if __name__ == "__main__":

    root = "./data/XHGNet/train-val"
    f_path = get_img_path_ls(root)

    init_img = cv2.imread(f_path[0])
    img_h, img_w, _ = init_img.shape

    imgs = np.zeros([img_h, img_w, 3, 1])

    print(f"img_h: {img_h}\nimg_w: {img_w}")
    print(f"num of imgs: {len(f_path)}")

    for idx, each_fpath in enumerate(tqdm(f_path)):
        img = cv2.imread(each_fpath)

        # img = cv2.resize(img, (img_h, img_w))

        # expand Batch dim
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

        imgs = imgs.astype(np.float32) / 255.0

        means, stdevs = [], []
        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()  # inplace flatten()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
        means.reverse()  # BGR --> RGB
        stdevs.reverse()

        if idx % 10 == 0:
            with open("result_of_mean_std.txt", "a+") as af:
                af.write(
                    f"Batch-{idx}: transforms.Normalize(normMean = {means}, normStd = {stdevs})\n"
                )

    print("normMean_rgb = {}".format(means))
    print("normStd_rgb = {}".format(stdevs))
    print(
        "Overall: transforms.Normalize(normMean = {}, normStd = {})".format(
            means, stdevs
        )
    )
