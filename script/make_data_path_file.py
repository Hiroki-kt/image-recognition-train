import glob
import os
import sys
import random


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
    print(len(train_list))
    print(len(val_list))
    return train_list, val_list


def write_txt(output_txt, file_list):
    with open(output_txt, 'w') as f:
        for d in file_list:
            f.write("%s\n" % d)


if __name__ == '__main__':
    # arg1: input directory path
    # arg2: output directory path
    args = sys.argv
    result = listup_files(args[1])
    train, val = train_val_sep(result)
    write_txt(args[2] + '/train.txt', train)
    write_txt(args[2] + '/val.txt', val)
