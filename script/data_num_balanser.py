import cv2
import numpy as np
import glob
import json
import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def reset_dict(dic):
    for i in dic:
        dic[i] = 0
    return dic


def voc_label_dist(data_path):
    key_open = open('./data_num.json', 'r')
    key_json = json.load(key_open)

    label_dist = reset_dict(key_json)
    for label_path in glob.glob(data_path + '/Annotations/*'):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for bb in root.findall('object'):
            label_dist[bb.find('name').text] += 1
        if not root.findall('object'):
            label_dist['no-obj'] += 1
    return label_dist


def check_destination(dest, output_path):
    labels = voc_label_dist(output_path)
    for i in dest:
        if dest[i] > labels[i]:
            return 1
        else:
            continue
    return 0


def check_labels(label_root, enough_labels):
    for bb in label_root.findall('object'):
        # print(i.text)
        label = bb.find('name').text
        if label in enough_labels:
            # print("skip")
            return 0
    return 1


def update_enough_labels(output_path, enough_labels):
    key_open = open('./data_num.json', 'r')
    key_json = json.load(key_open)

    labels = voc_label_dist(output_path)
    print("after", voc_label_dist(output_path))
    for label in labels:
        if label in enough_labels:
            continue
        if labels[label] >= key_json[label]:
            enough_labels.append(label)
    return enough_labels


def voc_copy_imgs(data_path, output_path):
    key_open = open('./data_num.json', 'r')
    key_json = json.load(key_open)

    origin_label_dist = voc_label_dist(data_path)
    # add_label_dist = reset_dict(origin_label_dist.copy())

    cp_count = 1
    enough_labels = []
    print("before", origin_label_dist)
    while check_destination(key_json, output_path):
        for label_path in glob.glob(data_path + '/Annotations/*'):
            label_tree = ET.parse(label_path)
            label_root = label_tree.getroot()
            if check_labels(label_root, enough_labels):
                ants_path = output_path + '/Annotations/' + \
                    Path(label_path).stem + '-'
                imgs_path = output_path + '/JPEGImages/' + \
                    Path(label_path).stem + '-'
                img_path = label_path.replace(
                    '.xml', '.jpg').replace('Annotations', 'JPEGImages')
                # print(img_path)
                if label_root.findall('object'):
                    label_tree.write(ants_path + str(cp_count) + '.xml')
                    shutil.copy(img_path, imgs_path + str(cp_count) + '.jpg')
                elif 'no-obj' not in enough_labels:
                    if voc_label_dist(output_path)['no-obj'] >= key_json['no-obj']:
                        label_tree.write(ants_path + str(cp_count) + '.xml')
                        shutil.copy(img_path, imgs_path +
                                    str(cp_count) + '.jpg')
        enough_labels = update_enough_labels(output_path, enough_labels)
        cp_count += 1
    print("finish", voc_label_dist(output_path))


if __name__ == '__main__':
    # arg1: input directory path
    # arg2: output directory path
    args = sys.argv
    voc_copy_imgs(args[1], args[2])
