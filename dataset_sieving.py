from pathlib import Path
from shutil import copy
import os


def mkdirs(overflow: str, underflow: str):
    """Create train and val dirs."""
    try: 
        if os.path.exists(overflow) is False:
            os.makedirs(overflow)
        if os.path.exists(underflow) is False:
            os.makedirs(underflow)
    except Exception as e:
        print(e)
        exit(-1)
    print("dirs created!")

def get_fname_list(path_list):
    """Convert PosixPath list into filename."""
    fname_list = []
    for fpath in path_list:
        fname = fpath.name
        fname_list.append(fname)
    return fname_list

def get_fpath_list(dpath: str):
    """Create filepath list."""
    return list(Path(dpath).rglob("*.jpg"))

def sieving(sieve_ls: list, feed_list: list):
    """sieving feed into over/under-flow."""
    overflow = []
    underflow = []
    for fpath in feed_list:
        if fpath.name in sieve_ls:
            overflow.append(fpath)
        else:
            underflow.append(fpath)
    print(f"len(train): {len(underflow)}\nlen(val): {len(overflow)}")
    return underflow, overflow


if __name__ == "__main__":

    # root PATH
    root = Path().joinpath("data", "XHGNet")

    train_val = str(root.joinpath("train-val").resolve())
    train = str(root.joinpath("train").resolve())
    val = str(root.joinpath("val").resolve())
    sieve = str(root.joinpath("val-XHG").resolve())

    # make dirs for train and val datasets
    mkdirs(overflow=val, underflow=train)

    # create flist, i.e., sieve and feed
    sieve_fpath_list = get_fpath_list(sieve)     
    sieve_fname_list = get_fname_list(sieve_fpath_list)

    feed_fpath_list = get_fpath_list(train_val)

    # sieving, i.e., split datasets into train and val according to val-XHG  
    train_fpath, val_fpath = sieving(sieve_ls=sieve_fname_list, feed_list=feed_fpath_list)

    # move datasets
    for index, image in enumerate(train_fpath):
        # copy tree to `train` datasets
        new_image = Path(train).joinpath(image.parts[-2], image.name)
        if os.path.exists(new_image.parent) is False:
            os.makedirs(new_image.parent)
        copy(image, new_image)
        print("train: \r[{}] processing [{}/{}]".format(image.parts[-2], index+1, len(train_fpath)), end="")  # processing bar
        print()

    # move datasets
    for index, image in enumerate(val_fpath):
        # copy tree to `train` datasets
        new_image = Path(val).joinpath(image.parts[-2], image.name)
        if os.path.exists(new_image.parent) is False:
            os.makedirs(new_image.parent)
        copy(image, new_image)
        print("val: \r[{}] processing [{}/{}]".format(image.parts[-2], index+1, len(val_fpath)), end="")  # processing bar
        print()

    print("processing done!")