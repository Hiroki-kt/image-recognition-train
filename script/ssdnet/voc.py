# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
# from trimming import settings


class VOCDataset(object):
    labels = None

    def __init__(self, root, year, subset):
        self.root = os.path.join(root, 'VOC' + year)

        path = os.path.join(self.root, 'ImageSets', 'Main', subset + '.txt')
        self.images = [line.strip() for line in open(path)]
        print("[%s] #images: %d" % (self.root, len(self.images)))

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return self.images[i]

    def image_path(self, i):
        return os.path.join(self.root, 'JPEGImages', self.images[i] + '.jpg')

    def image(self, i):
        """
        オリジナル。改造対象のため未使用。
        """
        return cv2.imread(
            os.path.join(self.root, 'JPEGImages', self.images[i] + '.jpg'),
            cv2.IMREAD_COLOR)

    @classmethod
    def get_object2(cls, elem, isfilter=True):
        """
        座標情報をbbox形式で返すバージョン。
        """
        # <object> tag
        name = elem.find('name').text
        if (isfilter):
            if (name not in VOCDataset.labels):
                # 対象オブジェクト情報でない場合はNoneを返す
                print("[get_object2] name: %s is skipped, because ai.settings.lables doesn't have it."
                      % (name))
                return None, None, name
        bndbox = elem.find('bndbox')
        # bbox = tuple(
        #     float(bndbox.find(t).text) - 1
        #     for t in ('xmin', 'ymin', 'xmax', 'ymax'))
        try:
            box = tuple(
                int(bndbox.find(t).text)
                for t in ('xmin', 'ymin', 'xmax', 'ymax'))
        except TypeError as e:
            dbginfo = tuple(
                bndbox.find(t).text
                for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            # print "[Skip] %s invalid format: %s" % ( self.dbg_apath, dbginfo )
            return None, None, name

        try:
            # find label ID by annotation data of /annotaion/object/name.text()
            label_id = VOCDataset.labels.index(name)
        except ValueError as e:
            label_id = None
        return label_id, VOCDataset.box2bbox(box), name

    @classmethod
    def get_object(cls, elem, valid=True):
        # <object> tag
        name = elem.find('name').text
        if (valid and (name not in VOCDataset.labels)):
            # 対象オブジェクト情報でない場合はNoneを返す
            # print "name: %s is skipped" % ( name )
            return None, None, name
        bndbox = elem.find('bndbox')
        # box = tuple(
        #     float(bndbox.find(t).text) - 1
        #     for t in ('xmin', 'ymin', 'xmax', 'ymax'))
        try:
            box = tuple(
                int(float(bndbox.find(t).text))
                for t in ('xmin', 'ymin', 'xmax', 'ymax'))
        except TypeError as e:
            dbginfo = tuple(
                bndbox.find(t).text
                for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            # print "[Skip] %s invalid format: %s" % ( self.dbg_apath, dbginfo )
            return None, None, name

        # find label ID by annotation data of /annotaion/object/name.text()
        if (VOCDataset.labels is None):
            # when VOCDataset.labels = None, always return class name
            label = name
        else:
            label = VOCDataset.labels.index(name)
        # print "get_object(): %s" % ( name )
        return label, box, name

    @classmethod
    def img2xml(cls, image_fn):
        bn = os.path.basename(image_fn)
        dirpath = os.path.dirname(image_fn)
        annotation_dir = os.path.abspath(
            os.path.join(dirpath, "../Annotations"))
        xmlfn = os.path.join(annotation_dir, os.path.splitext(bn)[0] + ".xml")
        return xmlfn

    @classmethod
    def get_annotations_by_image(cls, image_fn, valid=True):
        """
        image_fnに対応したAnnotation情報（bbox情報）を返す。
        """
        xmlfn = cls.img2xml(image_fn)
        if not (os.path.isfile(xmlfn)):
            print("[Error] %s does not exists..." % (xmlfn))
            # return []
            return np.empty((0, 4), float), np.empty((0, ), float)
        boxes, labels = VOCDataset.get_annotations(xmlfn, valid=valid)
        if (boxes):
            return np.array(boxes), np.array(labels)
        else:
            # 対象アノテーションが無い場合
            return np.empty((0, 4), float), np.empty((0, ), float)
        return boxes, labels

    @classmethod
    def bbox2box(self, bbox):
        """
        bbox(x, y, w, h)形式からbox(x, y, ox, oy)形式に変換
        """
        x, y, w, h = bbox
        return x, y, x+w, y+h

    @classmethod
    def box2bbox(self, bend):
        """
        box(x, y, ox, oy)形式からbbox(x, y, w, h)形式に変換
        """
        x, y, ox, oy = bend
        return x, y, ox-x, oy-y

    @classmethod
    def get_annotations2(cls, xmlfn, isfilter=True):
        """
        xmlfnに対応したAnnotation情報（bbox情報）を返す。
        """
        print("[get_annotations2] %s" % (xmlfn))
        tree = ET.parse(xmlfn)

        bboxes = list()
        labels = list()
        for child in tree.getroot():
            if not (child.tag == 'object'):
                # skip except object tag
                continue
            label_id, bbox, name = cls.get_object2(child, isfilter=isfilter)
            if (bbox is None):
                # settings.labelsで登録された対象オブジェクトでない場合はスキップ
                continue
            bboxes.append(bbox)
            labels.append(name)

            if (name == "person"):
                # for person landmarks
                for part in child.findall("part"):
                    label_id, bbox, name = cls.get_object2(
                        part, isfilter=isfilter)
                    if (bbox is None):
                        # settings.labelsで登録された対象オブジェクトでない場合はスキップ
                        continue
                    bboxes.append(bbox)
                    labels.append(name)
        return bboxes, labels

    @classmethod
    def get_annotations(cls, xmlfn, valid=True):
        """
        xmlfnに対応したAnnotation情報（bbox情報）を返す。
        """
        print("[get_annotations] %s" % (xmlfn))
        tree = ET.parse(xmlfn)

        boxes = list()
        labels = list()
        for child in tree.getroot():
            if not (child.tag == 'object'):
                # skip except object tag
                continue
            label, box, name = cls.get_object(child, valid=valid)
            if (box is None):
                # settings.labelsで登録された対象オブジェクトでない場合はスキップ
                continue
            boxes.append(box)
            labels.append(label)

            if (name == "person"):
                # for person landmarks
                for part in child.findall("part"):
                    label, box, name = cls.get_object(part)
                    if (box is None):
                        # settings.labelsで登録された対象オブジェクトでない場合はスキップ
                        continue
                    boxes.append(box)
                    labels.append(label)
        return boxes, labels

    def _annotations(self, xmlfn):
        """
        xmlfnに対応したAnnotation情報（bbox情報）を返す。
        """
        tree = ET.parse(xmlfn)

        boxes = list()
        labels = list()
        for child in tree.getroot():
            if not child.tag == 'object':
                # skip except object tag
                continue
            label, box, name = VOCDataset.get_object(child)
            if (box is None):
                # settings.labelsで登録された対象オブジェクトでない場合はスキップ
                continue
            boxes.append(box)
            labels.append(label)

            if (name == "person"):
                # for person landmarks
                for part in child.findall("part"):
                    label, box, name = VOCDataset.get_object(part)
                    if (box is None):
                        # settings.labelsで登録された対象オブジェクトでない場合はスキップ
                        continue
                    boxes.append(box)
                    labels.append(label)
        return boxes, labels

    def xml(self, i):
        return os.path.join(self.root, 'Annotations', self.images[i] + '.xml')

    def annotations(self, i):
        '''
        targetsが空でない場合、filter条件を満たすobject情報のみを出力する。
        '''
        apath = os.path.join(self.root, 'Annotations', self.images[i] + '.xml')
        self.dbg_apath = apath

        boxes, labels = self._annotations(apath)
        if (boxes):
            return np.array(boxes), np.array(labels)
        else:
            # 対象アノテーションが無い場合
            return np.empty((0, 4), float), np.empty((0, ), float)
