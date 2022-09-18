import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # if file_path exist, delete it firstly
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # set pseudo-random seed for reproductivity
    random.seed(0)

    # 10% validation datasets
    split_rate = 0.3

    # root/to/ImageFolder
    data_root = os.path.join(os.getcwd(), "..", "U-XHGNet")
    origin_image_path = os.path.join(data_root, "train-val")
    assert os.path.exists(origin_image_path), "path '{}' does not exist.".format(origin_image_path)

    image_class = [cls for cls in os.listdir(origin_image_path)
                    if os.path.isdir(os.path.join(origin_image_path, cls))]

    # mkdir(empty) for `train` datasets
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cls in image_class:
        mk_file(os.path.join(train_root, cls))

    # mkdir(empty) for `val` datasets
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cls in image_class:
        mk_file(os.path.join(val_root, cls))

    for cls in image_class:
        cls_path = os.path.join(origin_image_path, cls)
        images = os.listdir(cls_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_rate))  # random sampling by index
        for index, image in enumerate(images):
            if image in eval_index:
                # copy tree to `val` datasets
                image_path = os.path.join(cls_path, image)
                new_path = os.path.join(val_root, cls)
                copy(image_path, new_path)
            else:
                # copy tree to `train` datasets
                image_path = os.path.join(cls_path, image)
                new_path = os.path.join(train_root, cls)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cls, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
