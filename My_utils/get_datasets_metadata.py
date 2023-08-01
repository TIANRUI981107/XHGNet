import os
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageStat


def get_img_path_ls(root: str):
    """recursively glob JPG images."""
    ls = [str(p.resolve()) for p in Path(root).rglob("*.jpg")]
    return ls


def get_image_meta(image_id, mark="train-val"):
    """
    get image metadata.

    args:
        image_id: [str,] or [Posix path], /path/to/image
        dataset: [str,], types of datasets

    return:
        metadata of datasets [dict,]
    """
    image_src = image_id
    img = Image.open(image_src)
    extrema = img.getextrema()
    stat = ImageStat.Stat(img)

    meta = {
        "image": image_id,
        "dataset": mark,
        "R_min": extrema[0][0],
        "R_max": extrema[0][1],
        "G_min": extrema[1][0],
        "G_max": extrema[1][1],
        "B_min": extrema[2][0],
        "B_max": extrema[2][1],
        "R_avg": stat.mean[0],
        "G_avg": stat.mean[1],
        "B_avg": stat.mean[2],
        "R_std": stat.stddev[0],
        "G_std": stat.stddev[1],
        "B_std": stat.stddev[2],
        "height": img.height,
        "width": img.width,
        "format": img.format,
        "mode": img.mode,
    }
    return meta


def image_metadata(p_list, mark="train-val"):
    img_data = []

    pbar = tqdm(p_list, file=sys.stdout)
    for _, image_id in enumerate(pbar):
        img_data.append(get_image_meta(image_id, mark=mark))

    img_data_pd = pd.DataFrame(img_data)
    T_R_mean, T_G_mean, T_B_mean = (
        np.mean(img_data_pd["R_avg"]) / 255.0,
        np.mean(img_data_pd["G_avg"]) / 255.0,
        np.mean(img_data_pd["B_avg"]) / 255.0,
    )
    T_R_std, T_G_std, T_B_std = (
        np.mean(img_data_pd["R_std"]) / 255.0,
        np.mean(img_data_pd["G_std"]) / 255.0,
        np.mean(img_data_pd["B_std"]) / 255.0,
    )

    T_stat_df = pd.DataFrame(
        {
            "T_R_mean": T_R_mean,
            "T_G_mean": T_G_mean,
            "T_B_mean": T_B_mean,
            "T_R_std": T_R_std,
            "T_G_std": T_G_std,
            "T_B_std": T_B_std,
        },
        index=np.arange(1),
    )

    new_info_df = pd.concat([img_data_pd, T_stat_df], axis=1)
    return new_info_df


if __name__ == "__main__":
    dataset_name = "ONLINE"
    dataset_type = "train"
    root = f"../data/{dataset_name}/{dataset_type}"
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    f_path = get_img_path_ls(root)

    df_meta = image_metadata(f_path, mark=dataset_type)
    df_meta.to_csv(f"../metadata-{dataset_name}-{dataset_type}.csv", index=False)

    print("done!")
