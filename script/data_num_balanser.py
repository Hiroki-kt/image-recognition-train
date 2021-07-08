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
    old_img_path = "none"

    label_dist = reset_dict(key_json)
    for label_path in glob.glob(data_path + '/Annotations/*'):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for bb in root.findall('object'):
            label_dist[bb.find('name').text] += 1
        if not root.findall('object'):
            label_dist['no-obj'] += 1
    return label_dist


def check_destination(dest, now):
    for i in dest:
        if dest[i] > now[i]:
            return 1
        else:
            continue
    return 0


def save_img_xml(label_tree, output_path, label_path, cp_count):
    label_tree.write(output_path + '/Annotations/' +
                     Path(label_path).stem + '-' + str(cp_count) + '.xml')


def voc_copy_imgs(data_path, output_path):
    key_open = open('./data_num.json', 'r')
    key_json = json.load(key_open)
    old_img_path = "none"

    origin_label_dist = voc_label_dist(data_path)
    add_label_dist = reset_dict(origin_label_dist.copy())

    cp_count = 1
    print("before", origin_label_dist)
    while check_destination(key_json, add_label_dist):
        for label_path in glob.glob(data_path + '/Annotations/*'):
            label_tree = ET.parse(label_path)
            label_root = label_tree.getroot()
            for bb in label_root.findall('object'):
                # print(i.text)
                label = bb.find('name').text
                if add_label_dist[label] < key_json[label]:
                    add_label_dist[label] += 1
                else:
                    label_root.remove(bb)
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
            elif add_label_dist['no-obj'] < key_json['no-obj']:
                label_tree.write(ants_path + str(cp_count) + '.xml')
                shutil.copy(img_path, imgs_path + str(cp_count) + '.jpg')
                add_label_dist['no-obj'] += 1
        print("after", add_label_dist)
        cp_count += 1
    print("finish", voc_label_dist(output_path))


if __name__ == '__main__':
    # arg1: input directory path
    # arg2: output directory path
    args = sys.argv
    voc_copy_imgs(args[1], args[2])
