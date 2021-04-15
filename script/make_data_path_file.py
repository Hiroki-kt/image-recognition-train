import glob
import os
import sys


def listup_files(path):
    result = []
    for p in glob.glob(path + '/*'):
        # print(p)
        result.append(os.path.abspath(p))
    return result


def write_txt(output_txt, file_list):
    with open(output_txt, 'w') as f:
        for d in file_list:
            f.write("%s\n" % d)


if __name__ == '__main__':
    # arg1: input directory path
    # arg2: output txt file path
    args = sys.argv
    result = listup_files(args[1])
    write_txt(args[2], result)
