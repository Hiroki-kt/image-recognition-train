import glob
import os
import sys
import random


def listup_files(path):
    result = []
    for p in glob.glob(path + '/*'):
        # print(p.split(path + '/')[1].split('.jpg')[0])
        result.append((p.split(path + '/')[1].split('.jpg')[0]))
    return result


def write_txt(output_txt, file_list):
    with open(output_txt, 'w') as f:
        for d in file_list:
            f.write("%s\n" % d)


if __name__ == '__main__':
    # arg1: input directory path
    # arg2: output directory path
    args = sys.argv
    result = listup_files(args[1])
    write_txt(args[2] + '/default.txt', result)
