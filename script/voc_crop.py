import glob
import os
import sys
import random
import shutil


def listup_files(path):
    result = []
    for p in glob.glob(path + '/*'):
        # print(p)
        result.append(os.path.abspath(p))
    return result


def train_val_sep(path_list, val_rate=2):
    val_num = int(len(path_list) * 0.2)
    random.shuffle(path_list)
    train_list = path_list[val_num:]
    val_list = path_list[:val_num]
    for path in train_list:
        shutil.copy("", "")
    print('train:', len(train_list))
    print('test:', len(val_list))
    return train_list, val_list


def mkdir_voc_dir(path, test=True):
    os.makedirs(path + 'ImageSets/Main')


def cp_file(origin_file, ):
    print(origin_file)
    print(os.path.split(origin_file))
    shutil.copy(origin_file, "")


if __name__ == '__main__':
    # arg1: input directory path
    # arg2: output directory path
    args = sys.argv
    result = listup_files(args[1])
    train, val = train_val_sep(result)
