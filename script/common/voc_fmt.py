# -*- coding: utf-8 -*-
import sys
from lxml import etree as et
import argparse
import os
import csv
import time
import random


class VOCElem(object):
    def __init__(self, vocdir):
        self.path = vocdir
        self.jpeg_dir = os.path.join(self.path, "JPEGImages")
        self.annotation_dir = os.path.join(self.path, "Annotations")
        self.trainval_dir = os.path.join(self.path, "ImageSets/Main")

    def create_dirs(self):
        for dir in [self.jpeg_dir, self.annotation_dir, self.trainval_dir]:
            if not (os.path.isdir(dir)):
                print("Create Dir: %s" % (dir))
                os.makedirs(dir)

    def trainval_path(self):
        return os.path.join(self.trainval_dir, "trainval.txt")


class VOCMixIn(object):
    def vocdirs(self, basedir, name):
        """
        VOC形式の基本ディレクトリ構成を作成する。
        ディレクトリがあれば何もしない。
        """
        retdirs = []
        for d in ["ImageSets/Main/", "JPEGImages/", "Annotations/"]:
            dpath = os.path.join(basedir, name, d)
            retdirs.append(dpath)
            self.rmdirs(dpath)
            if (self.mkdirs(dpath)):
                print("Create Directory: %s" % (dpath))
        return retdirs

    def create_vocdirs(self, split=True):
        """
        Train/TestそれぞれのVOC形式の基本ディレクトリ構成を作成する。
        """
        for t in ["train", "test"]:
            if (split == False and t == "test"):
                # split = Falseの場合はtrainディレクトリしか作らない
                continue
            basedir = os.path.join(self.dir, t)
            self.vocdirs(basedir, self.name)

    def childxml(self, parent, key, value):
        """
        parentタグにkeyタグを付与する。
        valueがnot Noneの時はkeyタグのテキスト領域に定義される。
        """
        child = et.Element(key)
        if (value is not None):
            child.text = str(value)
        parent.append(child)
        return child

    def objectxml(self, root, clslbl, bbox):
        """
        <object>...</object>要素を1つ作成する。
        クラスラベルはclslblとする。
        """
        x, y, ox, oy = self.bbox2box(bbox)
        obj = self.childxml(root, "object", None)
        self.childxml(obj, "name", clslbl)
        self.childxml(obj, "truncated", "0")
        self.childxml(obj, "difficult", "0")
        bndbox = self.childxml(obj, "bndbox", None)
        self.childxml(bndbox, "xmin", x)
        self.childxml(bndbox, "ymin", y)
        self.childxml(bndbox, "xmax", ox)
        self.childxml(bndbox, "ymax", oy)

    def get_imageinfo(self, img_fpath):
        """
        画像: img_fpathの横幅、高さ、カラーチャンネル数を返す。
        """
        img = self.read_image(img_fpath)
        h, w = img.shape[:2]
        depth = img.shape[2]
        return w, h, depth

    def img2xml(self, image_fn):
        bn = os.path.basename(image_fn)
        dirpath = os.path.dirname(image_fn)
        annotation_dir = os.path.abspath(
            os.path.join(dirpath, "../Annotations"))
        xmlfn = os.path.join(annotation_dir, os.path.splitext(bn)[0] + ".xml")
        return xmlfn

    def save_vocxml(self, img_fpath, objects, outvoc):
        """
        Annotationディレクトリにxmlファイルを生成する。
        img_fpath: 入力画像ファイルパス
        objects: (label, bbox)の配列. [(label, bbox), (label, bbox), ...]
        """
        # get dataset folder name
        ds_name = os.path.abspath(outvoc.path).split("/")[-1]
        jpg_bn = os.path.basename(img_fpath)
        root = et.Element("annotation")
        self.childxml(root, "folder", ds_name)
        self.childxml(root, "filename", jpg_bn)
        source = self.childxml(root, "source", None)
        self.childxml(source, "database", "The %s Database" % (ds_name))
        self.childxml(source, "annotation", "%s" % (ds_name))
        self.childxml(source, "image", "JVIS")
        owner = self.childxml(root, "owner", None)
        self.childxml(owner, "flickrid", "JVIS Co., Ltd.")
        size = self.childxml(root, "size", None)
        width, height, depth = self.get_imageinfo(img_fpath)
        self.childxml(size, "width", width)        # 画像横幅
        self.childxml(size, "height", height)      # 画像高さ
        self.childxml(size, "depth", depth)        # 画像カラーチャンネル数
        self.childxml(root, "segmented", "0")
        # オブジェクト要素の作成
        for (label, bbox) in objects:
            # VOCはbbox形式ではなくbox形式で保存
            self.objectxml(root, label, bbox)

        # save as xml file
        xmlstr = et.tostring(root, pretty_print=True,
                             method='xml', encoding="utf-8").decode()
        xml_fpath = self.img2xml(img_fpath)
        xml_wfp = open(xml_fpath, "w")
        xml_wfp.write(xmlstr)
        xml_wfp.close()
        return xml_fpath

    def create_vocxml(self, test_or_train, d):
        basename = d["basename"]
        fn = "%s.jpg" % (basename)
        n = d["n"]
        root = et.Element("annotation")
        self.childxml(root, "folder", self.name)
        self.childxml(root, "filename", fn)
        source = self.childxml(root, "source", None)
        self.childxml(source, "database", "The %s Database" % (self.name))
        self.childxml(source, "annotation", "%s" % (self.name))
        self.childxml(source, "image", "JVIS")
        owner = self.childxml(root, "owner", None)
        self.childxml(owner, "flickrid", "JVIS Co., Ltd.")
        size = self.childxml(root, "size", None)
        self.childxml(size, "width", d["width"])        # 画像横幅
        self.childxml(size, "height", d["height"])      # 画像高さ
        self.childxml(size, "depth", d["depth"])        # 画像カラーチャンネル数
        self.childxml(root, "segmented", "0")
        if (d["data_type"] == "normal"):
            # 通常データの場合
            for face in d["face"]:
                self.objectxml(root, "head", face)
            for body in d["body"]:
                if (self.class_name):
                    # class_name指定がある場合はその名前を使う
                    self.objectxml(root, self.class_name, body)
                else:
                    self.objectxml(root, "person", body)
        elif ((d["data_type"] == "rescue") or (d["data_type"] == "trimming")):
            # RESCUE/TRIMMINGデータの場合:
            # CSVのbody, faceカラムいずれのデータもRESCUE（異常オブジェクト）クラスとして記載されている
            for rescue_obj in d["face"] + d["body"]:
                self.objectxml(root, d["label"], rescue_obj)
        # save as xml file
        xmlstr = et.tostring(root, pretty_print=True)
        xml_fpath = os.path.join(
            self.dir, test_or_train, self.name, "Annotations/%s.xml" % (basename))
        xml_wfp = open(xml_fpath, "w")
        xml_wfp.write(xmlstr)
        xml_wfp.close()
        return xml_fpath
