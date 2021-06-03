# -*- coding: utf-8 -*-
# from trimming import Trimming
from photo import Photo
# from ssdnet import MultiBoxEncoder, SSD300, SSD512, SSD300_v2, SSD512_v2, preproc_for_test
# from common.logger import get_task_logger
from datetime import datetime
from PIL import Image
import numpy as np
import sys
import os
import time

# logger = get_task_logger("trimming")


class Crop():
    """
    検出オブジェクトをCropした画像を取得するクラス
    """

    def __init__(self, options):
        self.options = options
        # ssdモデルのロード
        # self.models = self.load_ssdmodel()
        self.n_skip = 0
        self.n_data = 0
        # self.ssd_parts_scores = {}  # dummy: 最新実装に合わせるため

    def detect_for(self, clsname):
        """
        予め全モデルで予測を行い、
        clsnameで指定されたオブジェクトのみを出力する。
        """
        # if (photo):
        #     self.photo = photo

        # if (roicls):
        #     # roiclsで人体のいずれかの部位に限定
        #     roiobjs = []
        #     roi_bboxes = self.predict_by("body").get(roicls, [])
        #     for roi_bbox in roi_bboxes:
        #         roiobjs.append(roi_bbox)
        # else:
        #     # 写真全体をROIとする。roi=Noneはself.photoがオリジナル画像として扱われる。
        #     roiobjs = [None]
        roiobjs = [None]

        allobjs = {}
        for roi in roiobjs:
            # ROIの単位で検出
            roidicts = {}
            for mdl in self.get_modelinfo():
                # モデル別に検出（org画像の座標系でbbox出力）
                objs = self.predict_by(mdl["name"], roi=roi)
                roidicts.update(objs)
            # 全ROIの検出結果をクラス単位でマージ
            for k, v in roidicts.items():
                if not (k in allobjs):
                    allobjs[k] = v
                else:
                    allobjs[k] += v

        # clsnameで指定されたクラスのbboxを出力
        return allobjs.get(clsname, [])

    def detect_for2(self, clsname, roicls=None, photo=None, outroi=False):
        """
        予め全モデルで予測を行い、
        clsnameで指定されたオブジェクトのみを出力する。
        """
        if (photo):
            self.photo = photo

        if (roicls):
            # roiclsで人体のいずれかの部位に限定
            roiobjs = self.predict_by("body").get(roicls, [])
        else:
            # 写真全体をROIとする。roi=Noneはself.photoがオリジナル画像として扱われる。
            roiobjs = [None]

        allobjs, outrois = [], []
        for roi in roiobjs:
            # ROIの単位で検出
            roidicts = {}
            for mdl in self.get_modelinfo():
                # モデル別に検出（org画像の座標系でbbox出力）
                objs = self.predict_by(mdl["name"], roi=roi)
                roidicts.update(objs)
            if ((clsname in objs) and (len(objs[clsname]) > 0)):
                # 対象クラスオブジェクトが存在する場合のみ、元となったROIを保存
                outrois.append(roi)
                # 当該roiから複数のclsnameに関する対象オブジェクトが抽出された場合にも備え、配列を要素とする
                allobjs.append(objs[clsname])

        if (outroi):
            # clsnameで指定されたクラスのbboxと、そのroi領域を出力
            return allobjs, outrois
        else:
            # clsnameで指定されたクラスのbboxを出力
            return allobjs

    def crop_image(self, img, bbox, allow_zero=False):
        x, y, w, h = bbox
        if (not allow_zero) and ((x <= 0) or (y <= 0)):
            return None
        elif (allow_zero) and ((x < 0) or (y < 0)):
            return None
        croped_img = self.cut_rect(img, *bbox)
        if (self.options["pil"]):
            croped_img = self.cv2pil(croped_img)    # PIL形式に変換
        return croped_img

    def crop_images(self, img, bboxes):
        """
        bboxesで指定されたbboxに対応した画像を出力する。
        """
        logger.info("+++ [crop_images] start +++")
        images = []
        for crop_i, bbox in enumerate(bboxes):
            _x, _y, _w, _h = bbox
            logger.info("[crop_images] bbox: %s, %s, %s, %s" %
                        (_x, _y, _w, _h))
            croped_img = self.crop_image(img, bbox)
            if (croped_img is None):
                logger.info("[crop_images] <skip> invalid bbox:", bbox)
                continue
            images.append((crop_i, croped_img))
        logger.info("--- [crop_images] end ---")
        return images

    def saveimgs(self, images, img_fpath, odir):
        """
        img_fpathから抽出したimagesをodirに保存する
        """
        for i, (crop_i, img) in enumerate(images):
            fpath, ext = os.path.splitext(img_fpath)
            bn = os.path.basename(fpath)
            fn = "%s_%d%s" % (bn, i+1, ext)
            if not (os.path.exists(odir)):
                os.makedirs(odir)
            self.saveimg(img, odir, fn)

    def save_crop_images(self, img_fpath):
        # 写真のロード
        degree = 0      # 回転させない
        self.photo = Photo(img_fpath, 0, 0, {"rotate": degree})
        bboxes = self.detect_for(self.options["class"])

        if (self.options["n"] and (len(bboxes) > self.options["n"])):
            # 単身以外は除外
            self.n_skip += 1
            print("x [skip] #heads = %d: %s" % (len(bboxes), img_fpath))
            return

        # 修正後画像がある場合のみ画像生成
        bn = os.path.basename(img_fpath)
        mod_fpath = None
        if (self.options["mod"]):
            mod_fpath = os.path.join(self.options["mod"], bn)
            if not (os.path.isfile(mod_fpath)):
                self.n_skip += 1
                print("x [skip] %s does not exists..." % (mod_fpath))
                return

        self.n_data += 1
        # ORG画像（反射あり）から顔画像抽出
        images = self.crop_images(self.photo.src, bboxes)

        if (mod_fpath):
            # modified画像（反射あり）から顔画像抽出
            modimg = self.read_image(mod_fpath)
            mh, mw = modimg.shape[:2]
            if ((self.photo.w != mw) or (self.photo.h != mh)):
                print("x [Error] no same file-size pair: %s " % (mod_fpath))
                return
            mod_images = self.crop_images(modimg, bboxes)

        # resize
        if (self.options["shape"]):
            h, w = self.options["shape"].split("x")
            if not (h.isdigit() and w.isdigit()):
                print(
                    "[Error] no digits for --shape option. needs HEIGHTxWIDTH format...")
                return

            # アスペクト比維持してリサイズし、余った領域をpadding
            images = self.resize_images(images, (int(h), int(w)),
                                        pad=self.options["pad"], blur=self.options["blur"],
                                        aspect=self.options["pad"])
            if (mod_fpath):
                mod_images = self.resize_images(mod_images, (int(h), int(w)),
                                                pad=self.options["pad"], blur=self.options["blur"],
                                                aspect=self.options["pad"])

        # 修正前後の画像をそれぞれオブジェクト別にファイル書き出し
        self.saveimgs(images, img_fpath, self.options["output_org"])
        if (mod_fpath):
            self.saveimgs(mod_images, mod_fpath, self.options["output_mod"])

    def run(self):
        """
        バッチメイン
        """
        print("options:", end=' ')
        print(self.options)
        images = self.options["org"]
        if (os.path.isfile(images)):
            # ファイル指定の場合
            images = [images]
        elif (os.path.isdir(images)):
            # ディレクトリ指定の場合
            images = self.get_files(
                images, ext=[".jpg", ".jpeg", ".png", ".gif"])

        for i, img_fpath in enumerate(images):
            if (os.path.isfile(img_fpath)):
                print("[%d] %s" % (i, img_fpath))
                self.save_crop_images(img_fpath)
            else:
                print("[%d] <SKIP> no file: %s" % (i, img_fpath))

        print("#n_data: %d, #n_skip: %d" % (self.n_data, self.n_skip))
