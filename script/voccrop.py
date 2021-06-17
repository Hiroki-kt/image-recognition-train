# -*- coding: utf-8 -*-

from photo import Photo
from common.common import CommonMixIn
from crop import Crop
from common.voc_fmt import VOCMixIn, VOCElem
from datetime import datetime
from PIL import Image
import numpy as np
import os
import time
import csv
from argparse import ArgumentParser
import sys
sys.path.append("script")

# from trimming import Trimming
# from ssdnet import MultiBoxEncoder, SSD300, SSD512, SSD300_v2, SSD512_v2, preproc_for_test


class VOCCrop(VOCMixIn, CommonMixIn):
    """
    検出オブジェクトをCropした画像とそれに対応したVOCデータを生成するクラス
    """

    def __init__(self, options):
        super().__init__()
        self.options = options
        self.outvoc = VOCElem(self.options.output)
        self.outvoc.create_dirs()
        self.trainvals = []
        self.h, self.w = None, None
        self.skip_fn = self.options.skip    # csv skipリスト
        self.skip_fns = set([])
        self.n_skip = 0
        self.n_data = 0
        if (self.skip_fn is not None):
            skip_ifp = open(self.skip_fn, "rU")
            for i, rec in enumerate(csv.reader(skip_ifp, delimiter=",")):
                if (i == 0):
                    # header line
                    continue
                idx, fn, err_type, reason, _ = rec
                if (fn):
                    # スキップ対象ファイル名を登録（小文字に統一）
                    self.skip_fns.add(fn.lower())
            skip_ifp.close()
            print("[SKIP CSV] #entries: %d" % (len(self.skip_fns)))

        if (self.options.shape):
            height, width = self.options.shape.split("x")
            self.h, self.w = int(height), int(width)
            # print(type(self.h), type(self.w))
            # if not (self.h.isdigit() and self.w.isdigit()):
            #     print(
            #         "[Error] no digits for --shape option. needs HEIGHTxWIDTH format...")
            #     sys.exit(1)

    def crop_image(self, img, bbox):
        x, y, w, h = bbox
        if ((x <= 0) or (y <= 0)):
            return None
        croped_img = self.cut_rect(img, *bbox)
        if (self.options.pil):
            croped_img = self.cv2pil(croped_img)    # PIL形式に変換
        return croped_img

    def crop_images(self, img, bboxes):
        """
        bboxesで指定されたbboxに対応した画像を出力する。
        """
        images = []
        for crop_i, bbox in enumerate(bboxes):
            croped_img = self.crop_image(img, bbox)
            if (croped_img is None):
                print("[skip] invalid bbox:", bbox)
                continue
            images.append((crop_i, croped_img))
        return images

    def saveimgs(self, images, img_fpath, odir):
        """
        img_fpathから抽出したimagesをodirに保存する
        """
        newfpaths = []
        for i, (crop_i, img) in enumerate(images):
            fpath, ext = os.path.splitext(img_fpath)
            bn = os.path.basename(fpath)
            fn = "%s_%d%s" % (bn, i+1, ext)
            if not (os.path.exists(odir)):
                os.makedirs(odir)
            img_path = self.saveimg(img, odir, fn)
            newfpaths.append(img_path)
        return newfpaths

    def overlaped_ratio(self, bbox, bboxes):
        """
        bboxに対してbboxとオーバーラップした領域の割合が最大のものの割合と添字番号を返す。
        bboxes中にオーバーラップしているオブジェクトが１つも見付からない場合は(0.0, None)を返す。
        """
        max_overlaped, max_i = 0.0, None
        # print("bbox: ", bbox)
        for i, _bbox in enumerate(bboxes):
            area = self.intersection(bbox, _bbox)
            # print("_bbox: ", _bbox, area)
            overlaped_ratio = area / self.area(bbox)
            if (overlaped_ratio > max_overlaped):
                max_overlaped = overlaped_ratio
                max_i = i
        return max_overlaped, max_i

    def show_bboxes(self, img, img_fpath, objects):
        for (lbl, bbox) in objects:
            self.display_box(img, bbox, display_lbl=lbl)
        dir_path = os.path.dirname(img_fpath)
        bn, ext = os.path.splitext(os.path.basename(img_fpath))
        fn = "%s_dbg%s" % (bn, ext)
        self.saveimg(img, dir_path, fn)

    def conv_bbox2cropimg(self, crop_bboxes):
        """
        元画像基準のG.T.座標情報をcrop画像基準に変換する。
        G.T.情報のあるオブジェクト以外のcrop_bboxes(headオブジェクト)は除去される。
        返り値は、元画像から抽出されたCropID毎の配列として生成される。
        1つのcrop画像に対して、複数のオブジェクト座標が割り当たる可能性があるため、配列となっている。
        RETURN VALUE:
        {<crop_i1>: {"cropimg": <image>, "objects": [(lbl, bbox), (lbl, bbox), ...]},
         <crop_i2>: {"cropimg": <image>, "objects": [(lbl, bbox), (lbl, bbox), ...]},
         <crop_i3>: {"cropimg": <image>, "objects": []},
         ...
        }
        cropimgのみで、objectsが空配列のものは、座標データがない負例データを作成するという意味。
        """
        # 元画像基準のAnnotationを取得
        img = self.photo.src
        photo_boxes, photo_labels = self.get_answers(self.photo.image_fpath)
        if ((len(photo_boxes) == 0) or (len(photo_labels) == 0)):
            return {}
        photo_bboxes = [self.box2bbox(box) for box in photo_boxes]
        bboxes_in = {}
        selected_i, all_i = set([]), set(range(0, len(crop_bboxes)))
        print(photo_labels)
        print(photo_bboxes)

        for photo_bbox, photo_lbl in zip(photo_bboxes, photo_labels):
            # 座標情報のあるcrop_bboxオブジェクトのみ残される処理
            # 写真内のG.T.のobj座標がどのcrop画像に属するか？を選択
            # print(photo_bbox, photo_lbl)
            try:
                ratio, crop_i = self.overlaped_ratio(
                    photo_bbox, crop_bboxes)
                # print(ratio, crop_i)
            except ZeroDivisionError as e:
                continue
            if (ratio == 1.0):
                # crop画像内に完全に包含されているもののみ採用
                selected_i.add(crop_i)
                cx, cy, cw, ch = crop_bbox = crop_bboxes[crop_i]
                px, py, pw, ph = photo_bbox
                # crop画像基準にbboxを変換
                new_photo_bbox = px - cx, py - cy, pw, ph

                if not (crop_i in bboxes_in):
                    # 初回のみベースとなるcrop_imgを保存
                    crop_img = self.crop_image(img, crop_bbox)
                    if (crop_img is None):
                        # crop_imgが作成出来ない場合は保存対象外とする
                        print("[skip] invalid bbox [%d]:" %
                              (crop_i), new_photo_bbox)
                        continue
                    if (self.h and self.w):
                        # resize mode
                        crop_img = self.resize_image(crop_img, (self.h, self.w),
                                                     pad=self.options.pad, blur=self.options.blur,
                                                     aspect=self.options.pad)
                    bboxes_in[crop_i] = {"cropimg": crop_img, "objects": []}
                bboxes_in[crop_i]["objects"].append(
                    (photo_lbl, new_photo_bbox))
            elif (ratio > 0.0):
                # 一部オーバーレイしている場合
                # selected_i.add(crop_i)
                new_photo_bbox = []
                for i, _bbox in enumerate(crop_bboxes):
                    area = self.intersection(photo_bbox, _bbox)
                    # print("_bbox: ", _bbox, area)
                    overlaped_ratio = area / self.area(photo_bbox)
                    # print(overlaped_ratio)
                    if (overlaped_ratio < 0.2):
                        continue
                    if (overlaped_ratio != 1.0):
                        copy_bbox = list(photo_bbox)
                        if (photo_bbox[0] < _bbox[0]):
                            copy_bbox[0] = _bbox[0]
                            # print(copy_bbox)
                        # print(photo_bbox[0] + photo_bbox[2], _bbox[0] + _bbox[2] )
                        if (photo_bbox[0] + photo_bbox[2] > _bbox[0] + _bbox[2]):
                            # print(_bbox[0] + _bbox[2] - copy_bbox[0])
                            copy_bbox[2] = _bbox[0] + _bbox[2] - copy_bbox[0]
                        if (photo_bbox[0] + photo_bbox[2] <= _bbox[0] + _bbox[2]):
                            copy_bbox[2] = photo_bbox[0] + \
                                photo_bbox[2] - _bbox[0]
                    # print(copy_bbox)
                    new_photo_bbox.append((tuple(copy_bbox), i))
                # print(new_photo_bbox)
                for box in new_photo_bbox:
                    cx, cy, cw, ch = crop_bbox = crop_bboxes[box[1]]
                    px, py, pw, ph = box[0]
                    # crop画像基準にbboxを変換
                    tf_box = px - cx, py - cy, pw, ph

                    if not (box[1] in bboxes_in):
                        # 初回のみベースとなるcrop_imgを保存
                        crop_img = self.crop_image(img, crop_bbox)
                        if (crop_img is None):
                            # crop_imgが作成出来ない場合は保存対象外とする
                            print("[skip] invalid bbox [%d]:" %
                                  (box[1]), new_photo_bbox)
                            continue
                        if (self.h and self.w):
                            # resize mode
                            crop_img = self.resize_image(crop_img, (self.h, self.w),
                                                         pad=self.options.pad, blur=self.options.blur,
                                                         aspect=self.options.pad)
                        bboxes_in[box[1]] = {
                            "cropimg": crop_img, "objects": []}
                    bboxes_in[box[1]]["objects"].append(
                        (photo_lbl, tf_box))
        # print(selected_i)
        # for i in bboxes_in:
        #     print(bboxes_in[i]["objects"])

        # for noselected_i in list(all_i - selected_i):
        #     # 教師（座標）データに全くオーバーレイしていない（＝メガネかけていない）教師データを登録
        #     crop_bbox = crop_bboxes[noselected_i]
        #     crop_img = self.crop_image(img, crop_bbox)
        #     if (self.h and self.w):
        #         # resize mode
        #         crop_img = self.resize_image(crop_img, (self.h, self.w),
        #                                      pad=self.options.pad, blur=self.options.blur,
        #                                      aspect=self.options.pad)
        #     bboxes_in[noselected_i] = {"cropimg": crop_img, "objects": []}

        return bboxes_in

    def save_crop_images(self, img_fpath):
        # 写真のロード
        degree = 0      # 回転させない
        self.photo = Photo(img_fpath, 0, 0, {"rotate": degree})
        # crop_bboxesに対する添え字がIDとなる
        # crop_bboxes = self.detect_for(self.options["class"])    # headを取得
        # head内にG.T.情報のあるhead情報(crop_imgs)のみを抽出
        # todo: crop_bboxes = [(x, y, w, h)1, (x, y, w, h)2, ..., (x, y, w, h)5]
        crop_bboxes = [(1, 1535, 512, 512), (512, 1535, 512, 512),
                       (1024, 1535, 512, 512), (1536, 1535, 512, 512), (1935, 1535, 512, 512)]
        crop_imgs = self.conv_bbox2cropimg(crop_bboxes)
        # print(crop_imgs)
        self.n_data += 1

        # crop画像単位で画像保存＆Annotation生成
        for crop_i, crop_info in crop_imgs.items():
            objects = []
            cropimg = crop_info["cropimg"]
            objects = crop_info["objects"]
            fpath, ext = os.path.splitext(self.photo.image_fpath)
            bn = os.path.basename(fpath)
            print("bn: %s, crop_i: %s, ext: %s" % (bn, crop_i, ext))
            # crop_fnを画像としてsave
            crop_fn = "%s_%d%s" % (bn, crop_i+1, ext)
            crop_imgfpath = self.saveimg(
                cropimg, self.outvoc.jpeg_dir, crop_fn)
            # annotation xmlファイルをsave
            self.save_vocxml(crop_imgfpath, objects, self.outvoc)
            bn = os.path.splitext(os.path.basename(crop_imgfpath))[0]
            self.trainvals.append(bn)
        print()

    def write_trainval(self):
        """
        trainval.txtを新規作成suru.
        """
        wfn = self.outvoc.trainval_path()
        wfp = open(wfn, "w")
        wfp.write("\n".join(self.trainvals))
        wfp.close()
        print("[Done] Write #lines = %d" % (len(self.trainvals)))

    def run(self):
        """
        バッチメイン
        """
        # print("options:", end=' ')
        print(self.options)
        images = self.options.input
        if (os.path.isfile(images)):
            # ファイル指定の場合
            images = [images]
        elif (os.path.isdir(images)):
            # ディレクトリ指定の場合
            images = self.get_files(
                images, ext=[".jpg", ".jpeg", ".png", ".gif"])
        for i, img_fpath in enumerate(images):
            fn = os.path.basename(img_fpath).lower()
            if (fn in self.skip_fns):
                print("[SKIP] %s." % (fn))
                continue
            if (os.path.isfile(img_fpath)):
                print("[%d] %s" % (i, img_fpath))
                self.save_crop_images(img_fpath)
            else:
                print("[%d] <SKIP> no file: %s" % (i, img_fpath))
        # crop画像に対するtrainvalを作成
        self.write_trainval()

        print("#n_data: %d, #n_skip: %d" % (self.n_data, self.n_skip))


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-i', '--input', default=None, help='Name of model')
    argparser.add_argument('-b', '--mod', default=None,
                           help='Specify size of batch')
    argparser.add_argument('-e', '--class', default=None,
                           help='Specify number of epoch')
    argparser.add_argument('-dlc', '--output', default=None)
    argparser.add_argument('-po', '--outpu_mod',
                           default=False, help='Only execute predict.')
    argparser.add_argument('-a', '--arch', default="512")
    argparser.add_argument('-g', '--gpu', default="0", type=int)
    argparser.add_argument('-sh', '--shape', default=None)
    argparser.add_argument('-pd', '--pad', action='store_true', default=False)
    argparser.add_argument('-bl', '--blur', action='store_true', default=False)
    argparser.add_argument('-pi', '--pil', action='store_true', default=False)
    argparser.add_argument('-sk', '--skip', default=None)
    return argparser.parse_args()


if __name__ == '__main__':
    options = get_option()
    crop = VOCCrop(options)
    crop.run()
