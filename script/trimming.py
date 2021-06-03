# -*- coding: utf-8 -*-
from common.common import CommonMixIn
from common.logger import get_task_logger
# from chainer import serializers
# from ssdnet import MultiBoxEncoder, SSD300, SSD512, SSD300_v2, SSD512_v2, preproc_for_test
# from .exceptions import NoDetectException, ZoomLimitException, ShiftLimitException, OutOfTMLException, OutOfOrderException, SkipOfOrderException
# from .operation import Operation
from photo import Photo
# from .utils import get_trimmingposition_dir, make_trimmingposition_directory, delete_png_image
# from .trimming_fmt import TrimmingDataChecker
#from segments import get_seg_bboxes
from datetime import datetime
from copy import deepcopy
import numpy as np
# import chainer, dlib
import sys
import os
import time
import math
import json
import subprocess
# import requests
from trimming import settings

# logger = get_task_logger("trimming")


class SSDModel(object):
    def __init__(self):
        self.model = None               # モデルオブジェクト
        self.multibox_encoder = None    # MultiBoxEncoderオブジェクト
        self.th = None
        self.version = 1                # ssd network version


class Data(object):
    """
    1リクエスト単位の入出力情報を管理するクラス。
    """

    def __init__(self, op, photo):
        self.op = op
        self.photo = photo


class Trimming(CommonMixIn):
    def __init__(self, options):
        """
        オプションの取得。各種モデルのロード。
        """
        logger.info("[Trimming.__init__] options: %s" % (options))
        self.no = 0                     # trim_img連続実行回数
        self.image_dir = None
        self.options = options
        self.options.update({"arch": "512"})
        self.photo = None               # Photoインスタンス
        self.op = None                  # Operationインスタンス
        # 各種モデルのロード
        self.load_models()
        # debug info
        self.figure = None  # デバッグ表示用のpltオブジェクト
        self.checker = TrimmingDataChecker(settings.frame_image_path)

    def load_models(self):
        """
        ssdモデル、顔検出器、関節検出器をロードする
        """
        t1 = time.time()
        # ssdモデルのロード
        self.models = self.load_ssdmodel()
        # 顔検出・顔部位検出器のロード
        self.shape_predictor = self.load_facemodel()
        # 関節検出器のロード
        self.body_detector = None
        # self.body_detector = self.load_bodymodel()
        t2 = time.time()
        logger.info("--- [load_models] done: %0.2f (sec). ---" % (t2 - t1))

    def get_modelinfo(self):
        return settings.models

    def get_modelinfo_by_name(self, name):
        """ 対象(name)のモデル設定情報を取得する """
        info = {}
        for model in settings.models:
            if name == model.get("name", ""):
                info = model
        return info

    def load_ssdmodel(self):
        """
        3つのモデルを事前ロードする
        """
        models = {}
        # モデルのロード
        for mdl in settings.models:
            name = mdl['name']
            model_fpath = mdl['path']
            labels = mdl['labels']
            th = mdl['th']
            ver = mdl.get('version', 1)

            if (model_fpath):
                t1 = time.time()
                models[name] = SSDModel()
                models[name].th = th
                models[name].version = ver
                models[name].model = self._load_ssdmodel(
                    model_fpath, labels, ver)
                models[name].multibox_encoder = MultiBoxEncoder(
                    models[name].model)
                t2 = time.time()
                logger.info(
                    "[load_ssdmodel] <%s> model load done. %0.3f (sec)" % (name, t2 - t1))
        return models

    def _load_ssdmodel(self, model_fpath, labels, version):
        """
        1つのSSDモデルをロードする
        """
        if (self.options['arch'] == '300'):
            if (version == 1):
                model = SSD300(labels)
            elif (version == 2):
                model = SSD300_v2(labels)
        elif (self.options['arch'] == '512'):
            if (version == 1):
                model = SSD512(labels)
            elif (version == 2):
                model = SSD512_v2(labels)
        serializers.load_npz(model_fpath, model)
        if self.options['gpu'] >= 0:
            chainer.cuda.get_device(self.options['gpu']).use()
            model.to_gpu()
            logger.info("[_load_ssdmodel] GPU MODE: %d" %
                        (self.options['gpu']))
        return model

    def load_facemodel(self):
        """
        shape_predictor:    顔部位検出モデル
        をロードして返す。
        """
        shape_predictor = None
        if (settings.facial_keypoint_detector):
            t1 = time.time()
            shape_predictor = dlib.shape_predictor(
                settings.facial_keypoint_detector)
            t2 = time.time()
            logger.info(
                "[load_facemodel] <facial_keypoint> model load done. %0.3f (sec)" % (t2 - t1))
        return shape_predictor

    def load_bodymodel(self):
        """
        関節部位検出モデルのロード
        """
        from bodyjoint.keypoint import KeyPoint
        body_detector = None
        if (self.options['body']):
            # body detection
            use_gpu = self.options['gpu'] >= 0
            body_detector = KeyPoint(use_gpu)
        return body_detector

    def init_params_by_image(self, img_fpath, task_id, scene_id, imginfo):
        """
        写真単位で初期化されるパラメタ
        """
        self.photo = Photo(img_fpath, task_id, scene_id, imginfo)
        self.body_objs, self.allow_objs, self.deny_objs = [], [], []
        self.body_minimum_bbox, self.vbody_minimum_bbox, self.center = None, None, None
        self.allow_minimum_bbox, self.deny_minimum_bbox = None, None
        _x, _y, _zoom = self.get_default_value(task_id, img_fpath)
        self.op = Operation(_x, _y, _zoom, imginfo)
        self.algo = None
        self.chromakey_no = self.get_chromakey_no()
        self.dbg_center_type = ""
        self.dbg_center_base = ""
        # スコア込みの検出結果(modelに定められた閾値でfilterしたobjs) * scoreはnumpy.float32型
        # {'label': {(x, y, w, h): score}, ...} の形式でストアされる
        self.ssd_parts_scores = {}

    def get_default_value(self, task_id, img_fpath):
        _x, _y, _zoom = 0, 0, 1.0
        return _x, _y, _zoom

    def predict_by(self, model_name, roi=None):
        try:
            if (roi is not None):
                # ROI指定の場合は、現入力からROIエリアのみを切り取り
                self.photo.change_roi(roi)
            retval = self._predict_by(model_name)
            if (roi is not None):
                # 座標系をraw画像基準に変換
                for clslbl, bboxes in retval.items():
                    _bboxes = []
                    for bbox in bboxes:
                        # 座標が変わるのでスコア辞書の更新
                        score = self.ssd_parts_scores[clslbl][tuple(bbox)]
                        del self.ssd_parts_scores[clslbl][tuple(bbox)]
                        bbox = self.photo.org_coordinate(bbox)
                        _bboxes.append(bbox)
                        self.ssd_parts_scores[clslbl][tuple(bbox)] = score
                    retval[clslbl] = _bboxes
        finally:
            # 入力をオリジナル画像に戻す
            self.photo.change_org()
        return retval

    def _predict_by(self, model_name):
        """
        SSDによりbody, head検出を実行
        クラス毎にbboxの配列が可能のされた辞書を返す。
        {"person": [bbox11, bbox12, ...],
         "head": [bbox21, bbox22, ...], ...} フォーマットで返却。
        """
        t1 = time.time()
        if not (model_name in self.models):
            # 指定されたモデルがロードされていない場合
            return {}
        # 指定されたモデルをロード
        mdl = self.models[model_name]
        # 入力画像に対する前処理
        image_for_ssd = preproc_for_test(
            self.photo.src, mdl.model.insize, mdl.model.mean)
        t2 = time.time()
        if (self.options['gpu'] >= 0):
            # ndarray -> cupy arrayに変換
            image_for_ssd = chainer.cuda.to_gpu(image_for_ssd)
        # predict by SSD
        loc, conf = mdl.model(image_for_ssd[np.newaxis])
        t22 = time.time()
        loc = chainer.cuda.to_cpu(loc.data)
        conf = chainer.cuda.to_cpu(conf.data)
        t23 = time.time()
        try:
            boxes, labels, scores = mdl.multibox_encoder.decode(
                loc[0], conf[0], settings.iou_min, settings.conf_min)
        except ValueError as e:
            # skip this image
            return {}
        t3 = time.time()
        # 検出された不要なObjectを除外する
        boxes, labels, scores = self.filter_objs(mdl, boxes, labels, scores)
        # すでに一度判定済みか確認して、判定済みのときは(n_obj[LABEL]のカウントアップをしない)
        is_count_up = bool(self.photo.n_people() == 0)
        # 閾値以上の検出結果のみをフォーマッティング
        ssd_parts = {}
        for box, label, score in zip(boxes, labels, scores):
            box[:2] *= self.photo.src.shape[1::-1]
            box[2:] *= self.photo.src.shape[1::-1]
            box = box.astype(int)
            if (score > mdl.th):
                # 結果が閾値以上の場合:
                lblstr = mdl.model.labels[label]
                if is_count_up:
                    self.photo.count_up(lblstr)
                bbox = self.box2bbox(box)
                if not (lblstr in ssd_parts):
                    ssd_parts[lblstr] = []
                ssd_parts[lblstr].append(bbox)
                logger.info("[predict_by(%s)] (detect object) %0.3f, %s = %s"
                            % (model_name, score, lblstr, bbox.__str__()))
                # BBoxのスコア管理 # {'label': {(x, y, w, h): score}} の形式
                if not (lblstr in list(self.ssd_parts_scores.keys())):
                    self.ssd_parts_scores[lblstr] = {}
                self.ssd_parts_scores[lblstr][tuple(bbox)] = score
        t4 = time.time()
        logger.info("[predict_by(%s)] prepro: %0.3f (sec), predict: %0.3f (sec), postpro: %0.3f (sec)"
                    % (model_name, t2-t1, t3-t2, t4-t3))
        logger.debug("[predict_by(%s)] prepro: %0.3f (sec), predict1: %0.3f (sec), predict2: %0.3f, predict3: %0.3f, predict(ttl): %0.3f, postpro: %0.3f (sec)" % (
            model_name, t2-t1, t22-t2, t23-t22, t3-t23, t3-t2, t4-t3))
        return ssd_parts

    def predict_by_crops(self, model_name, bboxes):
        """
        対象モデル(model_name)にてオブジェクト判定を行う。
        但し、bboxにて囲われた領域をcropした画像が対象となる。
        cropに使うbbox集合(bboxes)を受け付ける。
        """
        predicted_objs = {}
        info = self.get_modelinfo_by_name(model_name)
        labels = info.get("labels", [])
        if labels is None:  # ラベル未定義の場合は判定の必要なし
            return predicted_objs
        for label in labels:
            predicted_objs[label] = []
        for bbox in bboxes:
            if bbox is None:
                continue
            # 予測座標がマイナスの場合があるので、0に補正
            _bbox = self.photo.to_collect_bbox(bbox)  # bboxのはみ出した部分を切り取る
            _predict_objs = self.predict_by(model_name, roi=_bbox)
            for predict_label, predict_objs in list(_predict_objs.items()):
                predicted_objs[predict_label] += predict_objs
        # logger.info("[predict_by_crops] %s" % predicted_objs)
        for label in labels:
            if len(predicted_objs[label]) == 0:  # 1つも検出しなかったものは出力しない
                del predicted_objs[label]
        return predicted_objs

    def filter_objs(self, mdl, boxes, labels, scores):
        """ 検出された同一ラベルで被り率が閾値以上のObject(面積が大きい方)を除外する """
        if settings.obj_ovl_select == 0:
            return boxes, labels, scores
        t1 = time.time()
        logger.info("[filter_objs] threshold: %s" % settings.obj_ovl_threshold)
        drop_indexes = []

        def adjust_obj(box, label):
            _box = deepcopy(box)
            _box[:2] *= self.photo.src.shape[1::-1]
            _box[2:] *= self.photo.src.shape[1::-1]
            _box = _box.astype(int)
            lblstr = mdl.model.labels[label]
            bbox = self.box2bbox(_box)
            return bbox, lblstr

        for index1, (box1, label1, score1) in enumerate(zip(boxes, labels, scores)):
            if (score1 <= mdl.th):
                # そもそもスコアが低いboxはとばす
                continue
            bbox1, lblstr1 = adjust_obj(box1, label1)
            for index2, (box2, label2, score2) in enumerate(zip(boxes, labels, scores)):
                if (index1 >= index2):
                    # [同じbox]と[すでにチェックしたbox]はとばす
                    continue
                if (score2 <= mdl.th):
                    # そもそもスコアが低いboxはとばす
                    continue
                bbox2, lblstr2 = adjust_obj(box2, label2)
                if (lblstr1 != lblstr2):
                    # ラベルが違う場合はチェックしない
                    continue
                logger.debug("[filter_objs] loop : (%s, %s)" %
                             (index1, index2))
                ovl2_base1 = self.box_ovl(bbox1, bbox2)
                ovl1_base2 = self.box_ovl(bbox2, bbox1)
                if (ovl2_base1 > settings.obj_ovl_threshold) or \
                        (ovl1_base2 > settings.obj_ovl_threshold):
                    # 少なくともどちらかのオーバラップ率が基準値を超えるとき
                    area1 = self.area(bbox1)
                    area2 = self.area(bbox2)
                    if bool(area1 > area2) == bool(settings.obj_ovl_select == 2):
                        # 面積の大きいbox1側(つまりindex1)を除外
                        drop_indexes.append(index1)
                    else:
                        # 面積の大きいbox2側(つまりindex2)を除外
                        drop_indexes.append(index2)
        logger.debug("[filter_objs] drop_indexes: %s" % (drop_indexes))
        # 除外するindex番号がきまったのでそれ以外をリストに詰め直して返す
        filtered_boxes, filtered_labels, filtered_scores = [], [], []
        for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if index in drop_indexes:
                continue
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)
        logger.info("[filter_objs] filtered objs: %s -> %s" %
                    (len(boxes), len(filtered_boxes)))
        t2 = time.time()
        logger.debug("[filter_objs] done: %0.2f (sec). ---" % (t2 - t1))
        return filtered_boxes, filtered_labels, filtered_scores

    def detect_ssdobjs(self):
        """
        人体、許可オブジェクトを取得する。
        """
        # {"person": [bbox11, bbox12, ...],
        #  "head": [bbox21, bbox22, ...], ...} フォーマットで検出結果を取得
        body_objs = self.predict_by("body")
        allow_objs = self.predict_by("allow")
        # allow_objs.update(self.predict_by("allow2")) # 追加の許可OBJ
        glasses_objs = self.predict_by_crops(
            "glass", body_objs.get("face", []))

        # todo: bodyとvbody（擬人オブジェクト）をどう扱うか？
        results = []
        for lblstr, cnt in self.photo.n_objs.items():
            results.append("#%s: %d" % (lblstr, cnt))
        logger.info("[detect_ssdobjs] %s" % (", ".join(results)))

        return body_objs, allow_objs, glasses_objs

    def detect_segmentobjs(self):
        """
        セグメンテーションを利用し、
        禁止オブジェクト(輪郭情報のみ)を取得する。
        """
        deny_contours = []
        self.deny_segment_fpath = None
        # 元画像を渡してsegmentation画像生成(禁止OBJ)
        deny_segment_fpath = self.get_segment_img_fpath()
        # deny_segment_fpath = self._get_segment_img_fpath()
        if deny_segment_fpath is None:
            return []
        elif os.path.exists(deny_segment_fpath):
            logger.info("[detect_segmentobjs] Found: %s" %
                        (deny_segment_fpath))
            self.deny_segment_fpath = deny_segment_fpath
            rotate = self.photo.imginfo.get("rotate", 0)
            deny_contours = self.get_contours(
                deny_segment_fpath, rotate=rotate, save_src=True)  # 輪郭抽出
        else:
            logger.info("[detect_segmentobjs] Not Found: %s" %
                        (deny_segment_fpath))
        logger.info("[detect_segmentobjs] n_deny: %d" % (len(deny_contours)))
        return deny_contours

    def get_center(self, upper=False):
        """
        人数によって変化する被写体中心を返す。
        (upper=Trueのときは、upper_centerもしくはupper_vbody_minimum_bboxを基準に算出する)
        """
        n_people = self.photo.n_people()
        if (n_people < settings.shift_horizontal_th):
            # 被写体中心の場合:
            logger.info(
                "[get_center] n_people: %d, center to center." % (n_people))
            self.dbg_center_type = "center to center"
            if (upper is True) and getattr(self, "upper_center", None):
                logger.info(
                    "[get_center] n_people: %d, center to center. (upper)" % (n_people))
                return getattr(self, "upper_center", None)
            return self.center
        else:
            # bbox中心の場合:
            logger.info(
                "[get_center] n_people: %d, vbody to center." % (n_people))
            self.dbg_center_type = "vbody to center"
            if (upper is True) and getattr(self, "upper_center", None):  # 上部判定ありの場合
                logger.info(
                    "[get_center] n_people: %d, vbody to center. (upper)" % (n_people))
                if getattr(self, "upper_vbody_minimum_bbox", None):
                    return self.centroid(self.upper_vbody_minimum_bbox)
                else:
                    return None
            if not getattr(self, "vbody_minimum_bbox", None):
                return None
            return self.centroid(self.vbody_minimum_bbox)

    def move_to_the_center(self, bbox, upper=False):
        """
        bboxをself.centerへ移動するためのシフト量xを算出する。
        """
        logger.info("+++ [move_to_the_center] start +++")
        fx, fy, fw, fh = bbox
        fcx = fx + (fw / 2)             # bbox中央
        tcx, tcy = self.get_center(upper=upper)    # 人数によって変換する被写体中心を取得
        # fcxをtcxへ移動するときの移動量を算出
        dx = tcx - fcx
        logger.info("+++ [move_to_the_center] done. dx = %d +++" % (dx))
        return dx

    def move_to_center(self, from_bbox, to_bbox):
        """
        from_bboxの中央をto_bboxの中央へ移動するためのシフト量xを算出する。
        （from_bbox基準）
        """
        tcx, tcy = self.centroid(to_bbox)            # のしめobjの重心
        fcx, fcy = self.centroid(from_bbox)          # TML枠の重心
        # from_bboxをto_bbox中央へ移動するときの移動量を算出
        dx = tcx - fcx
        return dx

    def zoom_to(self, delta, direction="v", is_log=True):
        """
        重心基準で、TML枠の1辺の半分を
        delta(px)で指定されたサイズ分増減たときのTML枠の拡縮率を返す。
        delta >= 0: delta (px) x 2の大きさ分、TML枠が拡大する。
        delta <  0: delta (px) x 2の大きさ分、TML枠が縮小する。
        """
        if is_log:
            logger.info("+++ [zoom_to(%d, %s)] start +++" % (delta, direction))
        # 入力画像を現在適用されている縮尺で取得。
        x, y, w, h = self.current_frame(is_log=is_log)
        # 重心中心の拡縮なので調整量を2倍しておく
        _delta = delta * 2
        if (direction == "v"):
            # タテ方向
            new = h + _delta
            ratio = new / float(h)
            if is_log:
                logger.info("[zoom_to (vertical-mode)] h: %d -> %d" % (h, new))
        else:
            # ヨコ方向
            new = w + _delta
            ratio = new / float(w)
            if is_log:
                logger.info(
                    "[zoom_to (horizontal-mode)] w: %d -> %d" % (w, new))
        if is_log:
            logger.info("[zoom_to] ratio = %f" % (ratio))
        return ratio

    def is_beyond_any_side(self, bbox, margin=settings.beyond_any_side_margin):
        """
        bboxの少なくとも一辺が画像端に及ぶ場合True、それ以外はFalse。
        """
        logger.info("+++ [is_beyond_any_side] start +++")
        # bboxの画像端チェック
        byd_sides = self.get_beyond_side(bbox, margin=margin)
        judge = byd_sides.any()
        logger.info("--- [is_beyond_any_side] done. ---")

        return judge

    def is_beyond_to_btm(self, bbox, margin=settings.beyond_any_side_margin):
        """
        bboxの下辺が画像端に及ぶ場合True、それ以外はFalse。
        """
        logger.info("+++ [is_beyond_any_side] start +++")
        # bboxの画像端チェック
        byd_sides = self.get_beyond_side(bbox, margin=margin)
        # [top, btm, lft, rht]でチェックしたい部分を１に
        judge = (byd_sides * [0, 1, 0, 0]).any()
        logger.info("--- [is_beyond_any_side] done. ---")

        return judge

    def get_beyond_side(self, bbox,
                        margin=settings.beyond_any_side_margin,
                        margins=None):
        """
        bboxが画像の四隅に及ぶ辺を特定する。
        四隅に及ぶ辺をTrueとする。
        marginsが与えられてない場合は、四方のマージンをすべてmarginで定義。
        marginsが与えられている場合は、margins(top, btm, lft, rht)に従い四方を個別定義する。
        (is_top, is_btm, is_left, is_right)
        """
        results = np.array([False] * 4)
        image_bbox = self.photo.image_bbox()
        i_x, i_y, i_ox, i_oy = self.bbox2box(image_bbox)
        x, y, ox, oy = self.bbox2box(bbox)
        if (margins is None):
            # marginsがNoneの場合はmarginを使う
            m_top, m_btm, m_lft, m_rht = margin, margin, margin, margin
        else:
            # marginsが与えられている場合は、四方のマージンを個別設定する
            m_top, m_btm, m_lft, m_rht = margins

        # minimum_areaが画像をはみ出す場合があるので左辺のマイナスはあり得る
        if ((x - i_x) < m_lft):
            # 左辺が画像端
            results[2] = True
        if ((i_ox - ox) < m_rht):
            # 右辺が画像端
            results[3] = True
        if ((y - i_y) < m_top):
            # 上辺が画像端
            results[0] = True
        if ((i_oy - oy) < m_btm):
            # 下辺が画像端
            results[1] = True

        logger.info("[get_beyond_side] margin: (%d, %d, %d, %d), result: (top, btm, lft, rht) = %s" % (
            m_top, m_btm, m_lft, m_rht, results))
        return results

    def is_single_wholebody(self):
        """
        単身＆全身写真の場合のみTrue、それ以外はFalseを返す。
        """
        return self.photo.is_single() and self.is_wholebody()

    def is_single_halfbody(self):
        """
        単身＆半身写真の場合のみTrue、それ以外はFalseを返す。
        """
        return self.photo.is_single() and (not self.is_wholebody())

    def is_couple_wholebody(self):
        """
        ２人＆全身写真の場合のみTrue、それ以外はFalseを返す。
        """
        return (self.photo.n_people() == 2) and self.is_wholebody()

    def is_couple_halfbody(self):
        """
        ２人＆半身写真の場合のみTrue、それ以外はFalseを返す。
        """
        return (self.photo.n_people() == 2) and (not self.is_wholebody())

    def is_wholebody(self):
        """
        単身に限らず全身・半身判定を行う。
        全身の場合True、半身の場合Falseを返す。
        """
        retval = False
        # faceの平均高を取得
        ave_w, ave_h = self.ave_bboxes(self.body_objs.get('face', []))
        # 縦横写真それぞれで閾値判定
        if (self.photo.is_portrait()):
            # タテ写真の場合:
            if (ave_h < settings.th_v_wholebody):
                retval = True
                logger.info(
                    "[is_wholebody] (portrait) True because of face height is small.")
        else:
            # ヨコ写真の場合:
            if (ave_h < settings.th_h_wholebody):
                retval = True
                logger.info(
                    "[is_wholebody] (landscape) True because of face height is small.")

        for person_bbox in self.body_objs.get('person', []):
            if (self.is_beyond_to_btm(person_bbox)):
                # bodyが画像境界付近の場合: 半身と判定
                retval = False
                logger.info(
                    "[is_wholebody] False because of is_beyond_any_side = True.")
                break
        logger.info("[is_wholebody] retval = %s" % (retval))
        return retval

    def get_minimum_area_for_body(self, body_objs, allow_objs, allow_vbody_cls=settings.vbody_cls):
        """
        複数の予測結果から被写体の構成（被写体エリア、中心座標）を決定し、
           minimum_bbox, center_point
        を返す。
        ここで最小被覆領域: minimum_bboxは、
        face, head, personを全て覆うunion_bboxのこと。
        """
        if (self.photo.n_people() >= 1):
            # 単身 or 集合写真の場合: faceの集合で中心座標を決定
            if ("face" in body_objs) and (self.photo.n_objs["face"] == self.photo.n_people()):
                cx, cy = self.get_centroid(body_objs['face'])
                centroid_bboxes = deepcopy(body_objs['face'])
                self.dbg_center_base = "face"
            elif ("head" in body_objs) and (self.photo.n_objs["head"] == self.photo.n_people()):
                # 顔が未検出の場合はheadで被写体中心を算出
                cx, cy = self.get_centroid(body_objs['head'])
                centroid_bboxes = deepcopy(body_objs['head'])
                self.dbg_center_base = "head"
            elif ("person" in body_objs) and (self.photo.n_objs["person"] == self.photo.n_people()):
                # 顔が未検出の場合はbodyで被写体中心を算出
                cx, cy = self.get_centroid(body_objs['person'])
                centroid_bboxes = deepcopy(body_objs['person'])
                self.dbg_center_base = "person"
            else:
                # 被写体中心が計算不能な場合は、処理不能例外
                raise NoDetectException(
                    "cannot compute a center point because of face, head and person are zero.")
        else:
            # 人体未検出の場合は、処理不能例外
            raise NoDetectException("cannot compute a center point.")
        center = (cx, cy)

        # 最小被覆領域の決定
        # personで拡張
        if ((self.photo.n_objs['person'] == 0) or (self.photo.n_objs['head'] == 0)):
            # 人体検出数が0の場合はトリミングできないので処理不能例外
            raise NoDetectException(
                "cannot compute minimum_bbox because of #person/#head is zero.")
        # masqによる判定領域内にpersonがなければ[調整トリミング]不可能
        if (not 'person' in body_objs):
            return {}, {}, None

        minimum_bbox = body_objs['person'][0]
        for person_bbox in body_objs.get('person', [])[1:]:
            minimum_bbox = self.union_bbox(minimum_bbox, person_bbox)
        # headで拡張
        for head_bbox in body_objs.get('head', []):
            minimum_bbox = self.union_bbox(minimum_bbox, head_bbox)
        # faceで拡張
        for face_bbox in body_objs.get('face', []):
            minimum_bbox = self.union_bbox(minimum_bbox, face_bbox)
        vbody_minimum_bbox = minimum_bbox
        # vbodyで拡張
        for lblstr, vbody_bboxes in allow_objs.items():
            # if ( lblstr in settings.vbody_cls ):
            if (lblstr in allow_vbody_cls):
                centroid_bboxes += deepcopy(vbody_bboxes)
                for vbody_bbox in vbody_bboxes:
                    vbody_minimum_bbox = self.union_bbox(
                        vbody_minimum_bbox, vbody_bbox)
                # # このクラスの重心を求める
                # vbody_center = v_cx, v_cy = self.get_centroid(vbody_bboxes)
                # _center = ((v_cx + cx) / 2, (v_cy + cy) / 2)
                # logger.debug("[get_minimum_area_for_body] %s -> %s" % (center.__str__(), _center.__str__()))
                # center = _center
        center = self.get_centroid(centroid_bboxes)

        logger.info("[get_minimum_area_for_body] body_minimum_bbox: %s, vbody_minimum_bbox: %s, center: %s"
                    % (minimum_bbox.__str__(), vbody_minimum_bbox.__str__(), center.__str__()))
        return minimum_bbox, vbody_minimum_bbox, center

    def get_minimum_area_for(self, objs, targets=None):
        """
        objsの検出結果に対してtargetsに該当するクラスのbboxを対象として最小被覆領域を作成
        targets = Noneの場合は、全クラスが対象となる。
        出力は最小被覆領域のbbox。１つも該当しない場合はNoneを返す。
        """
        minimum_bbox = None
        for clslbl, bboxes in objs.items():
            if ((targets is None) or (clslbl in targets)):
                # 該当クラスの場合
                for bbox in bboxes:
                    if (minimum_bbox is None):
                        minimum_bbox = bbox
                    else:
                        minimum_bbox = self.union_bbox(minimum_bbox, bbox)

        logger.info("[get_minimum_area_for] minimum_bbox: %s" %
                    (minimum_bbox.__str__()))
        return minimum_bbox

    def filter_outer_objs(self, objs, th=0.5):
        """
        TML枠にthの割合以上オーバーラップしているオブジェクトのみ残す。
        """
        retval = {}
        has_small_cls = False
        framing_bbox = self.photo.framing_bbox()
        for lblstr, bboxes in objs.items():
            ary = []
            if (lblstr in settings.shift_small_cls):
                # min_hshift, max_hshiftを小さめに設定するクラスが検出されたか？
                self.op.has_small_cls = True
            if (lblstr in settings.shift_0_cls):
                # min_hshift, max_hshiftを0に設定するクラスが検出されたか？
                self.op.has_0_cls = True
            for bbox in bboxes:
                if (self.box_ovl(bbox, framing_bbox) >= th):
                    ary.append(bbox)
                else:
                    logger.info("[filter_outer_objs] %s %s is removed. out of TML."
                                % (lblstr, bbox.__str__()))
            if (ary):
                retval[lblstr] = ary
        return retval

    def filter_allow_objs(self, allow_objs=None):
        """
        許可objをフィルターした結果を残す
        """
        # TML枠とOvl率が0.5以上の許可objのみ残す
        if allow_objs is None:
            allow_objs = self.allow_objs
        aobjs = self.filter_outer_objs(allow_objs)
        n_skips = 0
        retvals = {}
        for lblstr, bboxes in aobjs.items():
            if (lblstr in settings.exclude_allow):
                # allow obj除外設定の場合
                logger.info(
                    "[filter_allow_objs] %s is removed. exclude_allow obj." % (lblstr))
                n_skips += len(bboxes)
                continue
            retvals[lblstr] = bboxes
            self.photo.n_allows += len(bboxes)
            if (lblstr in settings.vbody_cls):
                self.photo.n_vbody_allows += len(bboxes)
        logger.info("[filter_allow_objs] n_skips: %d, n_allows: %d" %
                    (n_skips, self.photo.n_allows))
        return retvals

    def to_scored_objs(self, label, objs):
        """
        オブジェクトをスコア付きオブジェクトへ変換する。具体的には、
        objs = [(x, y, width, height), ...]]
        を
        objs = [((x, y, width, height), score), ...]
        とする。
        """
        _objs = []
        for obj in objs:
            # 救出したオブジェクト
            score = self.ssd_parts_scores[label][tuple(obj)]  # objはbbox形式のlist
            # ここでのscoreはnumpy.float32型なのでpython floatに変換する(json serializeするため)
            _objs.append((obj, float(score)))
        logger.info("[to_scored_objs] %s: %s" % (label, _objs))  # for dbg
        return _objs

    def cleanup_deny(self):
        """
        禁止オブジェクトが検出されなければセグメンテーション(png)画像を削除する
        """
        logger.info("[cleanup_deny] incoming.")
        if(len(self.deny_contours) == 0):
            if self.deny_segment_fpath is None:
                logger.info("[cleanup_deny] : %s" % (self.deny_segment_fpath))
            elif os.path.exists(self.deny_segment_fpath):
                logger.info("[cleanup_deny] remove: %s" %
                            (self.deny_segment_fpath))
                delete_png_image(self.deny_segment_fpath)
            else:
                logger.info("[cleanup_deny] does not exist: %s" %
                            (self.deny_segment_fpath))
        if self.deny_segment_fpath:
            logger.info("[cleanup_deny] is file -> %s" %
                        os.path.exists(self.deny_segment_fpath))
        logger.info("[cleanup_deny] outgoing.")

    def detect_objects(self):
        """
        最小被覆領域、被写体中心の算出。
        """
        self.body_objs, self.allow_objs, self.glasses_objs = self.detect_ssdobjs()
        # 禁止Obj. 出力は輪郭情報。輪郭情報はjson化できないため後続に渡せないので注意
        self.deny_contours = self.detect_segmentobjs()

        # 許可/禁止objをフィルターして確定する
        self.allow_objs = self.filter_allow_objs()
        self.deny_contours = self.filter_deny_objs()
        self.deny_objs = self.get_deny_objs(
            self.deny_contours)  # allow_objsと同じ形式(矩形領域)に
        # 禁止obj未検出(フィルタ後)の場合は、セグメンテーション画像(png)を削除する
        self.cleanup_deny()

        # dbに保存するAllowBBoxをフィルタ後に確定する, 信頼度も渡す
        self.db_allow_bbox = {}
        stored_objs = [self.body_objs, self.allow_objs,
                       self.glasses_objs]  # AllowBBoxに渡すオブジェクト
        for stored_obj in stored_objs:
            for label, objs in list(stored_obj.items()):
                if label not in list(self.db_allow_bbox.keys()):
                    self.db_allow_bbox[label] = []
                # スコア付きBBOXに
                self.db_allow_bbox[label] += self.to_scored_objs(
                    label, objs[:])
        # dbに保存するDenyBBoxをフィルタ後に確定する, 信頼度は存在しない為そのまま渡す
        self.db_deny_bbox = deepcopy(self.deny_objs)

        # 複数のface, head, personに関する最小被覆領域、被写体中心を算出
        self.body_minimum_bbox, self.vbody_minimum_bbox, self.center =\
            self.get_minimum_area_for_body(self.body_objs, self.allow_objs)

        # 許可・禁止objそれぞれの最小被覆領域を取得
        self.allow_minimum_bbox = self.get_minimum_area_for(self.allow_objs)
        self.deny_minimum_bbox = self.get_minimum_area_for(self.deny_objs)

        # 複数(3)人以上のとき: 最小被覆領域の上部のみで再判定->横シフトに利用
        self.masq_detect_objects()
        logger.info("--- [detect_objects] done. ---")

    def masq_detect_objects(self):
        """
        複数人のとき、最小被覆領域を参考に、上部最小被覆領域、上部被写体中心を算出。
        """
        if (not self.is_masq()):  # 人数が規定数以上か判定
            self.upper_body_objs = []
            self.upper_allow_objs, self.upper_deny_objs, self.upper_glasses_objs = [], [], []
            self.upper_body_minimum_bbox = {}
            self.upper_vbody_minimum_bbox = {}
            self.upper_center = None
            self.upper_allow_minimum_bbox = {}
            self.upper_deny_minimum_bbox = {}
            self.segment_human_bboxes = []
            logger.info("--- [masq_detect_objects] skip. ---")
            return False
        return self._masq_detect_objects()

    def _masq_detect_objects(self):
        logger.info("--- [_masq_detect_objects] start. ---")
        # MASQ前後を分ける
        self.upper_body_objs, self.upper_allow_objs, self.upper_deny_objs, self.upper_glasses_objs =\
            self.masq_detect_ssdobjs()
        # 許可objのをフィルターして確定する
        self.upper_allow_objs = self.filter_allow_objs(
            allow_objs=self.upper_allow_objs)

        # 複数のface, head, personに関する最小被覆領域、被写体中心を算出
        self.upper_body_minimum_bbox, self.upper_vbody_minimum_bbox, self.upper_center =\
            self.get_minimum_area_for_body(
                self.upper_body_objs, self.upper_allow_objs)
        # マスクド領域を作成(「写真」のupper_vbody_minimum_bbox下部までとする)
        imx, imy, imox, imoy = self.photo.image_bbox()
        if self.upper_vbody_minimum_bbox and (len(self.upper_vbody_minimum_bbox) == 4):
            masq_box = [imx, imy, imox, self.bbox2box(
                self.upper_vbody_minimum_bbox)[3]]
        else:
            masq_box = None  # upper被覆領域が見つからないときはマスクせずに判定
        # セグメンテーション判定&bbox化
        self.segment_human_bboxes = self.get_human_contour_bboxes(
            masq_box=masq_box)
        # セグメンテーションデータを利用して、最小被覆領域を拡張する
        if self.upper_body_minimum_bbox and self.segment_human_bboxes:
            # upper最小被覆領域とセグメンテーションbboxがあれば拡張を実施
            for human_bbox in self.segment_human_bboxes:
                self.upper_vbody_minimum_bbox = self.union_bbox(
                    self.upper_vbody_minimum_bbox, human_bbox)
        # 縮めすぎ抑制: 使わない
        # self.upper_vbody_minimum_bbox = self.adjust_upper_vbody_minimum_bbox()
        # 許可・禁止objそれぞれの最小被覆領域を取得
        self.upper_allow_minimum_bbox = self.get_minimum_area_for(
            self.upper_allow_objs)
        self.upper_deny_minimum_bbox = self.get_minimum_area_for(
            self.upper_deny_objs)
        logger.info("--- [_masq_detect_objects] done. ---")
        return True

    def current_iline(self):
        """
        トリミング適用後の現在のIラインを返す。
        """
        # デフォルトTML枠を取得
        fx, fy, fw, fh = self.photo.framing_bbox()
        # デフォルトIラインを取得
        (tx, ty), (bx, by) = self.photo.iline()
        # TML枠からの高さの比を算出
        t_ratio = (ty - fy) / float(fh)         # TML枠の高さに対するIライン上段の比
        b_ratio = (by - fy) / float(fh)         # TML枠の高さに対するIライン下段の比
        logger.info("[current_iline] t_ratio = %0.3f, b_ratio = %0.3f" %
                    (t_ratio, b_ratio))

        # 現在のTML枠を取得
        cfx, cfy, cfw, cfh = self.current_frame()
        rty = cfy + round(t_ratio * cfh)
        rby = cfy + round(b_ratio * cfh)
        # x座標はTML枠の中心
        rtx = rbx = cfx + round(cfw / 2.)
        logger.info("[current_iline] (%d, %d), (%d, %d)" %
                    (rtx, rty, rbx, rby))
        return (rtx, rty), (rbx, rby)

    def current_frame(self, is_log=True):
        """
        トリミング適用後の現在のフレーミング枠を返す。
        """
        # フレーミング枠の初期値を取得
        fx, fy, fw, fh = self.photo.framing_bbox()

        # シフトを適用
        # フレーミング枠側を動かす（シフトはTML枠の移動量なのでzoomスケールは関係ないので独立計算）
        dx = self.op.x
        dy = self.op.y
        fx += dx
        fy += dy
        # 拡縮を適用
        fx, fy, fw, fh = self.scale_bbox_on_center(
            (fx, fy, fw, fh), self.op.zoom)

        if is_log:
            logger.info("[current_frame] (%d, %d, %d, %d)" % (fx, fy, fw, fh))
        return fx, fy, fw, fh

    def current_frame_outer(self):
        """
        トリミング適用後の現在のフレーミング枠を返す。
        """
        # フレーミング枠の初期値を取得
        fx, fy, fw, fh = self.photo.framing_bbox_outer()

        # シフトを適用
        # フレーミング枠側を動かす（シフトはTML枠の移動量なのでzoomスケールは関係ないので独立計算）
        dx = self.op.x
        dy = self.op.y
        fx += dx
        fy += dy
        # 拡縮を適用
        fx, fy, fw, fh = self.scale_bbox_on_center(
            (fx, fy, fw, fh), self.op.zoom)

        logger.info("[current_frame_outer] (%d, %d, %d, %d)" %
                    (fx, fy, fw, fh))
        return fx, fy, fw, fh

    def padding_by_zoom(self):
        """
        人(vbody)が入るようにTML枠を縮小し余白を作成する
        """
        logger.info("+++ [padding_by_zoom] start +++")

        # 通常のズームによるパディングの場合:
        # 画像端まで及ぶ辺はパディングしない
        margin_top, margin_btm, margin_lft, margin_rht = self.get_margin_around(
            self.vbody_minimum_bbox)
        padding, _di = 0, ""
        if (self.photo.is_portrait()):
            if (self.is_wholebody()):
                # タテ・全身写真: left/right=3px+, bottom=20px+ 余白あける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["portrait"]["wholebody"]
                padding_btm, padding_h = 0, 0
                if (margin_btm < p_btm):
                    padding_btm = (p_btm - margin_btm)
                    logger.info("[padding_by_zoom] padding_btm = %d" %
                                (padding_btm))
                    _di = "btm"
                if ((margin_lft < p_lft) or (margin_rht < p_rht)):
                    padding_h = p_lft - min([margin_lft, margin_rht])
                    if (margin_lft < margin_rht):
                        _di = "lft"
                    else:
                        _di = "rht"
                    logger.info("[padding_by_zoom] padding_h = %d" %
                                (padding_h))
                padding = max([padding_btm, padding_h])
            else:
                # タテ・半身写真: left/right=3px+, top=20px+ 余白あける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["portrait"]["halfbody"]
                padding_top, padding_h = 0, 0
                if (margin_top < p_top):
                    padding_top = (p_top - margin_top)
                    logger.info("[padding_by_zoom] padding_top = %d" %
                                (padding_top))
                    _di = "top"
                if ((margin_lft < p_lft) or (margin_rht < p_rht)):
                    padding_h = p_lft - min([margin_lft, margin_rht])
                    if (margin_lft < margin_rht):
                        _di = "lft"
                    else:
                        _di = "rht"
                    logger.info("[padding_by_zoom] padding_h = %d" %
                                (padding_h))
                padding = max([padding_top, padding_h])
        else:
            if (self.is_wholebody()):
                # ヨコ・全身: top=17+ 余白空ける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["landscape"]["wholebody"]
                if (margin_top < p_top):
                    padding = (p_top - margin_top)
                    _di = "top"
            else:
                # ヨコ・半身: top=17+ 余白空ける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["landscape"]["halfbody"]
                if (margin_top < p_top):
                    padding = (p_top - margin_top)
                    _di = "top"
        logger.info("[padding_by_zoom] padding = %d" % (padding))

        if (padding):
            # TML枠を拡大してパディングを作る
            ratio = self.zoom_to(padding, direction="v")
            self.op.update_zoom(ratio, text="padding: %s" % (_di))
            logger.info(
                "[padding_by_zoom] normal zoom padding = %d, zoom = %0.3f" % (padding, ratio))
        else:
            logger.info("[padding_by_zoom] no zoom.")
        logger.info("--- [padding_by_zoom] done. ---")

    def zoom_in(self, inner_bbox, text="",
                margin=settings.beyond_any_side_margin,
                margins=None):
        """
        拡大により内側にあるinner_bboxをTML枠外に出す
        """
        logger.info("+++ [zoom_in] start +++")
        # TML枠からはみ出しているかチェック。
        # はみ出している場合はマイナス値が入る。
        _margins = margin_top, margin_btm, margin_lft, margin_rth = \
            self.get_margin_around(
                inner_bbox, margin=margin, margins=margins, reverse=True)

        # 余白の最大値を求める
        margin_max_i = _margins.argmax()
        margin_max = _margins[margin_max_i]
        logger.info("[zoom_in] margin_max: %d" % (margin_max))
        if (margin_max > 0):
            # -sys.maxintとなった辺はプラスにはならないので対象外となっている
            # はみ出している場合: TML枠に入る縮小率を求める
            if ((margin_max == margin_top) or (margin_max == margin_btm)):
                # 縦方向の調整
                ratio = self.zoom_to(margin_max, direction="v")
            else:
                # 横方向の調整
                ratio = self.zoom_to(margin_max, direction="h")
            # zoom処理保存
            self.op.update_zoom(ratio, text="%s: %s" %
                                (text, self.pos2direction(margin_max_i)))
        else:
            logger.info("[zoom_in] no zoom.")

        logger.info("[zoom_in] current_frame: %s" %
                    (self.current_frame().__str__()))
        logger.info("--- [zoom_in] done. ---")

    def zoom_out(self, inner_bbox, text="",
                 margin=settings.beyond_any_side_margin,
                 margins=None):
        """
        縮小によりinner_bboxをTML枠内に入れる
        """
        logger.info("+++ [zoom_out] start +++")
        # TML枠からはみ出しているかチェック。
        # はみ出している場合はマイナス値が入る。
        _margins = margin_top, margin_btm, _, _ = self.get_margin_around(inner_bbox,
                                                                         margin=margin, margins=margins)

        # 余白の最小値を求める
        margin_min_i = _margins.argmin()
        margin_min = _margins[margin_min_i]
        logger.info("[zoom_out] margin_min: %d" % (margin_min))
        if (margin_min < 0):
            # sys.maxintとなった辺はマイナスにはならないので対象外となっている
            # はみ出している場合: TML枠に入る縮小率を求める
            if ((margin_min == margin_top) or (margin_min == margin_btm)):
                # 縦方向の調整
                ratio = self.zoom_to(-margin_min, direction="v")
            else:
                # 横方向の調整
                ratio = self.zoom_to(-margin_min, direction="h")
            # zoom処理保存
            self.op.update_zoom(ratio, text="%s: %s" %
                                (text, self.pos2direction(margin_min_i)))
        else:
            logger.info("[zoom_out] no zoom.")

        logger.info("[zoom_out] current_frame: %s" %
                    (self.current_frame().__str__()))
        logger.info("--- [zoom_out] done. ---")

    def pos2direction(self, i):
        if (i == 0):
            return "top"
        elif (i == 1):
            return "btm"
        elif (i == 2):
            return "lft"
        elif (i == 3):
            return "rht"
        return ""

    def zoom_for_person(self):
        """
        最小被覆領域をTML枠内にパディング付きで入れる。
        """
        logger.info("+++ [zoom_for_person] start +++")
        try:
            # 縮小で最小被覆領域をTML枠内に入れる
            self.zoom_out(self.vbody_minimum_bbox, text="zo person")
            # 縮小でパディングを作る
            self.padding_by_zoom()
        except (ZoomLimitException, ShiftLimitException) as e:
            # 許容値越え(TML枠外)例外の場合
            # todo: 例外キーを保存
            pass
        logger.info("--- [zoom_for_person] done. ---")

    def get_extended_frame_for_allow(self):
        """
        1/3ルール適用して拡張したTML枠を返す。
        """
        # はみ出し領域を無視して内包しているTML辺から画像端までの1/3の距離を取得
        side_margins = self.get_allow_margin()
        # TML枠に1/3領域を追加
        top, btm, lft, rht = side_margins
        fx, fy, fw, fh = self.current_frame()
        ex_framing_bbox = round(
            fx - lft), round(fy - top), round(fw + lft + rht), round(fh + top + btm)
        logger.info("[get_extended_frame_for_allow] extended frame: %s" % (
            ex_framing_bbox.__str__()))
        return ex_framing_bbox

    def get_allow_margin(self):
        """
        現在のTML枠に対して、四方の1/3 marginを算出する。
        retval: [top, btm, lft, rht]
        """
        # 元画像に対してフレーミング枠の四方のmarginを求める
        side_margins = np.array(self.margin_around(
            self.current_frame(), self.photo.image_bbox()))
        # 正数のみ残し、負数は0とする。
        side_margins = side_margins * (side_margins > 0)
        # それぞれの1/3マージンを取得
        side_margins = side_margins / 3
        logger.info("[get_allow_margin] side_margins (top, btm, lft, rht) = (%d, %d, %d, %d)"
                    % (side_margins[0], side_margins[1], side_margins[2], side_margins[3]))
        return side_margins

    def zoom_for_allowobjs(self):
        """
        許可objをTML枠内に入れる。
        """
        logger.info("+++ [zoom_for_allowobjs] start +++")
        try:
            if not (self.allow_objs):
                return
            # 1/3ルールのために画像端から2/3をマージン領域とする
            allow_margins = self.get_allow_margin() * 2
            # 許可objに対する最小被覆領域でzoom outトリミング
            # 重心基準でzoom outトリミングするので、marginは最大値を与える。
            # self.zoom_out(self.allow_minimum_bbox, margin=allow_margins.max(), text="zo allow")
            self.zoom_out(self.allow_minimum_bbox,
                          margins=allow_margins, text="zo allow")
        except (ZoomLimitException, ShiftLimitException) as e:
            # 許容値越え(TML枠外)例外の場合
            # todo: 例外キーを保存
            pass
        logger.info("--- [zoom_for_allowobjs] done. ---")

    def zoom(self):
        """
        ズーム処理によりTML枠内へ入れ、パディングを作る。
        """
        logger.info("+++ [zoom] start +++")
        if (self.photo.is_portrait()):
            # タテ写真の場合
            if (self.is_wholebody()):
                # 全身の場合
                self.zoom_for_person()          # 最小被覆領域をTML枠内に入れる
            else:
                # 半身の場合
                self.zoom_for_person()          # 最小被覆領域をTML枠内に入れる
                # logger.info("[zoom] skip zoom_for_person() because of is_wholebody = False.")
            # 許可objを枠内に入れる
            self.zoom_for_allowobjs()
        else:
            # ヨコ写真の場合
            if (self.is_wholebody()):
                # 全身の場合
                self.zoom_for_person()          # 最小被覆領域をTML枠内に入れる
            else:
                # 半身の場合
                self.zoom_for_person()          # 最小被覆領域をTML枠内に入れる
                # logger.info("[zoom] skip zoom_for_person() because of is_wholebody = False.")
            # 許可objを枠内に入れる
            self.zoom_for_allowobjs()
        logger.info("--- [zoom] done. ---")

    def in_tml_with_dbg(self, inner_bbox, frame=None, margin=settings.beyond_any_side_margin):
        """
        inner_bboxがframing_bbox枠内か？検査。
        枠内であればTrue、そうでなければFalse。
        ただし、画像端に及んでいる辺はTrue扱いとする。
        """
        logger.info("+++ [in_tml] start +++")
        margins = self.get_margin_around(
            inner_bbox, frame=frame, margin=margin)
        if ((margins >= 0).all()):
            # 四方全てがTML枠内に収まっている場合
            retval = True
        else:
            retval = False

        logger.info("--- [in_tml] done. margins: %s, retval = %s" %
                    (margins, retval))
        return retval, margins

    def in_tml(self, inner_bbox, frame=None, margin=settings.beyond_any_side_margin):
        """
        inner_bboxがframing_bbox枠内か？検査。
        枠内であればTrue、そうでなければFalse。
        ただし、画像端に及んでいる辺はTrue扱いとする。
        """
        logger.info("+++ [in_tml] start +++")
        margins = self.get_margin_around(
            inner_bbox, frame=frame, margin=margin)
        if ((margins >= 0).all()):
            # 四方全てがTML枠内に収まっている場合
            retval = True
        else:
            retval = False

        logger.info("--- [in_tml] done. margins: %s, retval = %s" %
                    (margins, retval))
        return retval

    def margin_around(self, inner_bbox, outer_bbox):
        """
        外側の矩形領域:outer_bboxに対し、内側:inner_bboxの上下左右方向の余白を求める。
        はみ出しはマイナスとなる。
        """
        logger.info("+++ [margin_around] start +++")
        ox, oy, ow, oh = outer_bbox
        ix, iy, iw, ih = inner_bbox
        logger.info("[margin_around] outer_bbox = (%d, %d, %d, %d)" %
                    (ox, oy, ow, oh))
        logger.info("[margin_around] inner_bbox = (%d, %d, %d, %d)" %
                    (ix, iy, iw, ih))

        pad_top = iy - oy
        pad_btm = (oy + oh) - (iy + ih)
        pad_lft = ix - ox
        pad_rht = (ox + ow) - (ix + iw)

        margin = [pad_top, pad_btm, pad_lft, pad_rht]

        logger.info(
            "--- [margin_around] done. (top, btm, lft, rht) = %s ---" % (margin.__str__()))
        return margin

    def get_margin_around(self, inner_bbox, frame=None,
                          margin=settings.beyond_any_side_margin,
                          margins=None, reverse=False):
        """
        inner_bboxとTMLの内包関係をチェックし、
        TML枠からの距離を返す。
        TML枠内の辺は正数、枠外の辺は負数で返る。
        ただし、inner_bboxの四方の辺のうち、画像端に及ぶ辺はmaxint扱いとする。
        ＊reverse=Trueのときは、画像端に及ぶ辺は-maxint扱いとする。
        @return: [top, btm, left, right] (px)
        """
        logger.info("+++ [get_margin_around] start +++")
        if (frame is None):
            # 現在のフレーミング枠を取得
            framing_bbox = self.current_frame()
        else:
            # 引数で渡された枠をフレーミング枠とする
            framing_bbox = frame
        # inner_bboxのどの辺が画像端まで及んでいるか？チェック
        byd_sides = self.get_beyond_side(
            inner_bbox, margin=margin, margins=margins)
        # TML枠に対するinner_bboxの余白（はみ出しはマイナス）を求める
        logger.info("[get_beyond_around] inner_bbox = (%d, %d, %d, %d)" %
                    (inner_bbox[0], inner_bbox[1], inner_bbox[2], inner_bbox[3]))
        _margins = np.array(self.margin_around(inner_bbox, framing_bbox))

        # 画像端に及ぶ辺は無視する
        for i in range(4):
            if (byd_sides[i]):
                # 画像端まで及んでいる場合はTML枠からはみ出してないことにする
                if reverse:
                    _margins[i] = -1 * sys.maxsize
                else:
                    _margins[i] = sys.maxsize

        logger.info("[get_margin_around] margin (top, btm, lft, rht) = (%d, %d, %d, %d)"
                    % (_margins[0], _margins[1], _margins[2], _margins[3]))
        return _margins

    def get_shift_padding(self, inner_bbox,
                          margin=settings.beyond_any_side_margin,
                          margins=None):
        """
        inner_bboxがTML枠内に収まるためのシフト量(dx, dy)を算出。
        収めるのが無理な場合はOutOfTMLException例外を発生し処理確定しない。
        """
        logger.info("+++ [get_shift_padding] start +++")
        _margins = self.get_margin_around(inner_bbox,
                                          margin=margin, margins=margins)
        if ((_margins >= 0).all() and (_margins < sys.maxsize).all()):
            # TML枠内に収まっている場合
            logger.info(
                "--- [get_shift_padding] done. inner_bbox is already in flame. ---")
            return 0, 0

        # 枠内に収められるか？チェック（画像端判定された辺はsys.maxintとなっているため無視される）
        top, btm, lft, rht = _margins
        # 上下もしくは左右にはみ出しておりシフトで対応できない場合は例外
        if ((top < 0) and (btm < 0)):
            # 上下にはみ出しており対応不可の場合
            logger.info(
                "--- [get_shift_padding] OutOfTMLException. top and btm are out of frame. ---")
            raise OutOfTMLException("out of both top and bottom template.")
        elif ((lft < 0) and (rht < 0)):
            # 左右にはみ出しており対応不可の場合
            logger.info(
                "--- [get_shift_padding] done. OutOfTMLException. lft and btm rht out of frame. ---")
            raise OutOfTMLException("out of both left and right template.")

        dx, dy = 0, 0
        if (top < 0):
            # topがはみ出し
            dy = top
        elif (btm < 0):
            # bottomがはみ出し
            dy = abs(btm)

        if (lft < 0):
            # leftがはみ出し
            dx = lft
        elif (rht < 0):
            # rightがはみ出し
            dx = abs(rht)
        logger.info(
            "--- [get_shift_padding] done. dx = %d, dy = %d. ---" % (dx, dy))
        return dx, dy

    def padding_by_shift_for_vbody(self):
        """
        人体の最小被覆領域: self.vbody_minimum_bboxをシフトでTML枠に収める。
        はみ出す場合やマスク切れをおこす場合は例外出力。
        """
        logger.info("+++ [padding_by_shift_for_vbody] start +++")
        # 最小被覆領域でTML枠に対してはみ出し判定し、シフト量を計算
        dx, dy = self.get_shift_padding(self.vbody_minimum_bbox)
        if ((dx == 0) and (dy == 0)):
            logger.info(
                "--- [padding_by_shift_for_vbody] done. no padding by shift because of both dx and dy is 0. ---")
            return 0, 0
        fx, fy, fw, fh = self.current_frame()
        fx += dx
        fy += dy
        margins = np.array([])
        # minimum vbodyが枠内に入るか？検証
        is_in_tml, margins = self.in_tml_with_dbg(
            self.vbody_minimum_bbox, frame=(fx, fy, fw, fh))
        if (is_in_tml):
            # シフト操作でTML枠内に収まる場合
            out_x, out_y, out_w, out_h = self.current_frame_outer()
            out_x += dx
            out_y += dy
            margin_min_i = margins.argmin()
            _di = self.pos2direction(margin_min_i)
            if not (self.is_miss(frame=(out_x, out_y, out_w, out_h))):
                # マスク切れが発生しない場合:
                self.op.update_shift(dx, dy, text="pad_vbody: %s " % (_di))
                logger.info(
                    "[padding_by_shift_for_vbody] Shift OK. (dx, dy) = (%d, %d)" % (dx, dy))
                # TML枠の移動ベクトルを返す。
                return (dx, dy)
            else:
                # TML枠内で画像の欠損がある場合（マスク切れが発生する場合）:
                logger.info(
                    "[padding_by_shift_for_vbody] ShiftLimitException. is_miss() = True. direction: %s" % (_di))
                # 故意に例外を発生
                self.op.add_shift_exception(
                    dx, dy, "pad_vbody mask loss: %s" % (_di))
        else:
            # シフトでTML枠に入れられない場合は例外であきらめる
            logger.info(
                "[padding_by_shift_for_vbody] ShiftLimitException. is_miss() = False. Shift (dx, dy) = (%d, %d) is out of TML." % (dx, dy))
            # 故意に例外を発生
            self.op.add_shift_exception(
                dx, dy, "pad_vbody: not in TML vbody by shift.")

    def padding_by_shift_for_allow(self, body_dx, body_dy):
        """
        許可objの最小被覆領域: allow_minimum_bboxをシフトでTML枠に収める。
        その際、1/3ルールで枠に収める。
        はみ出す場合は例外出力。
        """
        logger.info("+++ [padding_by_shift_for_allow] start +++")
        if (self.allow_minimum_bbox is None):
            # 許可objが無い場合は処理しない
            logger.info(
                "--- [padding_by_shift_for_allow] done. no allow obj ---")
            return (0, 0)

        # 1/3ルールのために画像端から2/3をマージン領域とする
        allow_margins = self.get_allow_margin() * 2
        # 1/3ルール適用のときのみ、TML枠内にシフトで入れる
        dx, dy = self.get_shift_padding(
            self.allow_minimum_bbox, margins=allow_margins)
        if ((dx == 0) and (dy == 0)):
            logger.info(
                "--- [padding_by_shift_for_allow] no shift because of both dx and dy is 0. ---")
            return (0, 0)

        # angle((dx, dy), (body_dx, body_dy)) > 90度の場合は、許可objは動かさない
        p1 = (dx, dy)
        p2 = (body_dx, body_dy)
        _angle = self.angle(p1, p2)
        logger.info("[padding_by_shift_for_allow] _angle: %d" % (int(_angle)))
        if (_angle > 90):
            # 許可objを動かすとvbodyがはみ出す場合: 例外発生し未処理で終了
            logger.info(
                "--- [padding_by_shift_for_allow] ShiftLimitException. angle = %d ---" % (_angle))
            # 故意に例外を発生
            self.op.add_shift_exception(
                dx, dy, "pad_allow: angle: %d" % (int(_angle)))

        # 動かした後も、はみ出しチェックしてはみ出していたら元のシフト量に戻す
        fx, fy, fw, fh = self.current_frame()
        fx += dx
        fy += dy
        # dx, dyによりallowは1/3ルールで補正対象となった辺がTML枠に入っているので、
        # vbody_minimum_bboxが新たにはみ出してなければ採用する。
        if (self.in_tml(self.vbody_minimum_bbox, frame=(fx, fy, fw, fh))):
            # シフト操作してもTML枠内に収まる場合
            self.op.update_shift(dx, dy, text="pad_allow")
            # TML枠の移動ベクトルを返す。
            logger.info(
                "--- [padding_by_shift_for_allow] Shift OK. (dx, dy) = (%d, %d) ---" % (dx, dy))
            return (dx, dy)
        else:
            # シフトでTML枠に入れられない場合は例外であきらめる
            logger.info(
                "--- [padding_by_shift_for_allow] ShiftLimitException exception occureed. ---")
            self.op.add_shift_exception(
                dx, dy, "pad_allow: not in TML vbody by shift")

    def shift_vertical(self):
        """
        上下の余白が1:1.5+の比率となる最小シフト量を計算する。
        """
        logger.info("+++ [shift_vertical] start +++")
        fx, fy, fw, fh = self.current_frame()           # TML枠
        mx, my, mw, mh = self.body_minimum_bbox         # bodyの最小被覆領域（vbodyは対象外）
        top = my - fy                                   # TML枠上辺から頭上までの余白
        btm = (fy + fh) - (my + mh)                     # TML枠下辺から足下までの余白
        if (self.is_vshift_ok(top, btm) == False):
            # v-shift対象外の場合
            logger.info(
                "[shift_vertical] SKIP because of is_vshift_ok = False.")
            return

        if (self.is_wholebody()):
            # 全身写真の場合
            ratio = top / float(btm)    # optical value = 1.5
            if (ratio > 1.5):
                # top:btm = 1.5+:1.0の場合
                # do-nothing
                logger.info(
                    "[shift_vertical] no shift because of top:btm = %0.2f:1." % (ratio))
            else:
                # top:btm = 1.5-:1.0の場合
                # TML枠を下方向にシフト
                # 3/5でtop側を求めることで移動量は最小化される
                all_padding = top + btm
                _top = math.ceil((all_padding * 3) / 5.)      # 1.5+を保証するため切り上げ
                _btm = all_padding - _top
                dy = top - _top
                # シフト量の反映（x方向は動かさないのでdx = 0）
                self.op.update_shift(0, dy, text="v")
                logger.info("[shift_vertical] dy = %d" % (dy))
        logger.info("--- [shift_vertical] done. ---")

    def both_out_horizontal(self, margin):
        """
        get_margin_around()の出力: marginを入力として、
        水平方向の両端が基準TML枠に対してはみ出しているか？を判定する。
        片方のみの場合はFalseとする。
        """
        top, btm, lft, rht = margin
        retval = False
        if (((lft < 0) or (lft == sys.maxsize)) and ((rht < 0) or (rht == sys.maxsize))):
            retval = True
        logger.info("[both_out_horizontal] retval: %s" % (retval))
        return retval

    def is_hshift_ok(self, dx):
        """
        水平方向のシフトが許容値範囲内であればTrueを返す。
        """
        retval = False
        if ((self.op.min_hshift < dx) and (dx < self.op.max_hshift)):
            retval = True
        else:
            # 水平シフト許容値範囲外の場合:
            # 故意に例外を発生
            self.op.add_shift_exception(dx, 0, "h1")
        return retval

    def is_vshift_ok(self, top, btm):
        """
        v-shift対象の場合はTrue、対象外の場合Falseを返す。
        """
        bx, by, bw, bh = self.body_minimum_bbox
        retval = True
        # TML枠外チェック
        if (top < 0):
            logger.info(
                "[is_vshift_ok] False because of top < 0 (top = %d)" % (top))
            retval = False
        elif (btm < 0):
            logger.info(
                "[is_vshift_ok] False because of btm < 0 (btm = %d)" % (btm))
            retval = False
        elif (bh < settings.min_body_h):
            # minimum bodyが400px未満の場合: v-shift対象外とする
            logger.info(
                "[is_vshift_ok] False because of minimum_body_height = %d < 400" % (bh))
            retval = False
        return retval

    def set_shift_minmax(self):
        """
        shift()からコールされる。
        人数や特定の許可Objクラス, 全身/半身, 縦/横写真の条件によって、
        水平シフト量最大値を変更する。
        ここではシーン特有のトリミング処理のシフト量変更はしない。
        """
        if (self.op.has_small_cls):
            # 小さめ設定の許可Objが検出されているとき
            self.op.set_hshift(settings.min_small_hshift,
                               settings.max_small_hshift)
            logger.info("[set_shift_minmax] (has_small_cls) %s <= hshift <= %s" %
                        (settings.min_small_hshift, settings.max_small_hshift))
        elif (self.op.has_0_cls):
            # 水平シフト0の許可Objが検出されているとき
            self.op.set_hshift(settings.min_0_hshift, settings.max_0_hshift)
            logger.info("[set_shift_minmax] (has_0_cls) %s <= hshift <= %s" %
                        (settings.min_0_hshift, settings.max_0_hshift))
        elif (self.photo.n_people() >= settings.shift_horizontal_th):
            # 3人以上のとき
            self.op.set_hshift(settings.min_with_horizontal,
                               settings.max_with_horizontal)
            logger.info("[set_shift_minmax] (peoples >= %s) %s <= hshift <= %s" %
                        (settings.shift_horizontal_th, settings.min_with_horizontal, settings.max_with_horizontal))
        elif self.is_couple_wholebody():
            # 2人&全身のとき
            self.op.set_hshift(settings.min_2wholebody_hshift,
                               settings.max_2wholebody_hshift)
            logger.info("[set_shift_minmax] (is_couple_wholebody) %s <= hshift <= %s" %
                        (settings.min_2wholebody_hshift, settings.max_2wholebody_hshift))
        elif self.is_couple_halfbody():
            # 2人&半身のとき
            self.op.set_hshift(settings.min_2halfbody_hshift,
                               settings.max_2halfbody_hshift)
            logger.info("[set_shift_minmax] (is_couple_halfbody) %s <= hshift <= %s" %
                        (settings.min_2halfbody_hshift, settings.max_2halfbody_hshift))
        else:
            logger.info("[set_shift_minmax] hshift not change")

    def is_masq(self):
        """
        写真に写っている人数が規定数(masq_threshold)以上かどうか判定する。
        規定数以上のときTrueを返す。
        # ドレス対策: 最小被覆領域の上部n[%]またはm[px]のみで判定した結果を水平シフトに利用する為
        """
        if (self.photo.n_people() >= settings.masq_threshold):
            return True
        else:
            return False

    def shift_horizontal(self):
        """
        水平シフト処理。
        3人以下の場合: 被写体中心を中央にシフトする。
        4人以上の場合: 最小被覆領域を中央にシフトする。
        """
        logger.info("+++ [shift_horizontal] start +++")
        logger.info("is_masq(): %s" % self.is_masq())
        framing_bbox = fx, fy, fw, fh = self.current_frame()   # TML枠
        if (self.photo.n_people() >= settings.shift_horizontal_th):
            dx = self.move_to_the_center(framing_bbox, upper=self.is_masq())
        else:
            dx = self.move_to_the_center(framing_bbox)
        # 人数が一定数のときは、最小被覆領域の上部のみで判定した結果で中心を計算: ドレス対策
        if (self.is_masq() and self.upper_vbody_minimum_bbox):
            vbody_minimum_bbox = self.upper_vbody_minimum_bbox
        else:
            vbody_minimum_bbox = self.vbody_minimum_bbox
        margin = self.get_margin_around(
            vbody_minimum_bbox, frame=(fx, fy, fw, fh))
        _both_out_horizontal = self.both_out_horizontal(margin)
        _is_hshift_ok = self.is_hshift_ok(dx)
        if ((_both_out_horizontal == False) and _is_hshift_ok):
            # 両端がはみ出しておらず、許容値以内の場合:
            fx += dx
            # 構図優先度はvbody > allowなので、vbodyのみで検証する
            # vbodyとallowのOR領域で判断する？
            margin = self.get_margin_around(
                vbody_minimum_bbox, frame=(fx, fy, fw, fh))
            # シフトするとはみ出す場合は、シフト出来る所までずらす
            if (margin[2] < 0 and (abs(dx) >= abs(margin[2]))):
                # 移動後leftがはみ出している場合
                _dx = dx + margin[2]
                _dx = _dx / 2     # 左右均等となるように移動量を半分に
                logger.info(
                    "[shift_horizontal] adjust shift to border: dx = %d -> %d" % (dx, _dx))
                dx = _dx
            elif (margin[3] < 0 and (abs(dx) >= abs(margin[3]))):
                # rightがはみ出している場合
                _dx = (dx - margin[3])
                _dx = _dx / 2     # 左右均等となるように移動量を半分に
                logger.info(
                    "[shift_horizontal] adjust shift to border: dx = %d -> %d" % (dx, _dx))
                dx = _dx

            if (dx):
                # 水平方向への移動が発生する場合
                try:
                    self.op.update_shift(dx, 0, text="h1")
                    logger.info("[shift_horizontal] centering. dx = %d" % (dx))
                except ShiftLimitException as e:
                    can_maxshift = False
                    if (dx > 0):
                        # 右方向へ動かして許容値越えの場合
                        if (margin[3] > dx):
                            # マージンあればmaxまで移動
                            dx = self.op.max_hshift - 1
                            can_maxshift = True
                    else:
                        # 左方向へ動かして許容値越えの場合
                        if (margin[2] < dx):
                            # マージンあればmaxまで移動
                            dx = self.op.min_hshift + 1
                            can_maxshift = True
                    if (can_maxshift and dx):
                        # 許容値最大で登録（この時は例外発生しない）
                        self.op.update_shift(dx, 0, text="h ex")
                        logger.info(
                            "[shift_horizontal] max/min centering. dx = %d" % (dx))
            else:
                logger.info("[shift_horizontal] no shift. dx = 0.")

        logger.info("--- [shift_horizontal] done. ---")

    def _shift(self):
        """
        シフト処理により配置バランスを取る（センタリング＆上下余白比率調整）
        """
        self.set_shift_minmax()
        # シフト通常処理
        if (self.photo.is_portrait()):
            # タテ写真:
            if (self.is_wholebody()):
                # 全身写真:
                self.shift_vertical()
                self.shift_horizontal()
            else:
                # 半身写真:
                self.shift_horizontal()
        else:
            # ヨコ写真:
            if (self.is_wholebody()):
                # 全身写真:
                self.shift_horizontal()
            else:
                # 半身写真:
                self.shift_horizontal()

    def shift(self):
        """
        シフト処理により配置バランスを取る（センタリング＆上下余白比率調整）
        """
        logger.info("+++ [shift] start +++")
        try:
            self._shift()
        except (ZoomLimitException, ShiftLimitException) as e:
            # 許容値越え(TML枠外)例外の場合
            # todo: 例外キーを保存
            pass
        except (OutOfTMLException) as e:
            # TML枠内に収まらなかった場合
            # todo: 例外キーを保存
            pass
        logger.info("--- [shift] done. ---")

    def is_miss(self, frame=None):
        """
        TML枠（外）に対して画像が欠損していればTrue。そうでなければFalse。
        """
        img_box = ix, iy, iox, ioy = self.bbox2box(self.photo.image_bbox())
        if (frame):
            framing_box_outer = fx, fy, fox, foy = self.bbox2box(frame)
        else:
            framing_box_outer = fx, fy, fox, foy = self.bbox2box(
                self.current_frame_outer())
        logger.info("[is_miss] image box (x, y, ox, oy) = %s" %
                    (img_box.__str__()))
        logger.info("[is_miss] frame box [outer] (x, y, ox, oy) = %s" %
                    (framing_box_outer.__str__()))

        if ((fx < 0) or (fy < 0)):
            # 入力画像に対してTML枠が左上にはみ出している場合
            logger.info(
                "[is_miss] ret = True. over top-left of image. (fx, fy) = (%d, %d)" % (fx, fy))
            return True
        elif ((fox > iox) or (foy > ioy)):
            # 入力画像に対してTML枠が右下にはみ出している場合
            logger.info("[is_miss] ret = True. over bottom-right of image. (fox, foy) = (%d, %d)"
                        % (fox, foy))
            return True
        # 欠損が無い場合
        logger.info(
            "--- [is_miss] done. ret False. not miss image is in frame. ---")
        return False

    def adjust_by_zoom(self):
        """
        TML枠内の欠損領域をTML枠の縮小で埋める。
        どの方向をadjustしたか？を返す('top', 'btm', 'lft', 'rht')。
        補正しなかった場合は、空語('')を返す。
        """
        ibbox = ix, iy, iox, ioy = self.bbox2box(self.photo.image_bbox())
        logger.info(
            "+++ [adjust_by_zoom] start. image_bbox: %s +++" % (ibbox.__str__()))
        framing_bbox = self.current_frame()
        framing_bbox_outer = fx, fy, fw, fh = self.current_frame_outer()
        fx, fy, fox, foy = self.bbox2box(framing_bbox_outer)

        if ((fx < 0) or (fy < 0)):
            # 入力画像に対してTML枠が左上にはみ出している場合
            logger.info(
                "[adjust_by_zoom] over top-left of image. (fx, fy) = (%d, %d)" % (fx, fy))
            # TML枠を拡大して欠損部分を入れる
            if (fx < 0):
                ratio = self.zoom_to(fx, direction="h")
                self.op.update_zoom(ratio, text="adj_z: lft")
                direction = "lft"
            if (fy < 0):
                ratio = self.zoom_to(fy, direction="v")
                self.op.update_zoom(ratio, text="adj_z: top")
                direction = "top"
        elif ((fox > iox) or (foy > ioy)):
            # 入力画像に対してTML枠が右下にはみ出している場合
            logger.info(
                "[adjust_by_zoom] over bottom-right of image. (fox, foy) = (%d, %d)" % (fox, foy))
            # TML枠を拡大して欠損部分を入れる
            _dx = fox - iox
            if (_dx > 0):
                ratio = self.zoom_to(-_dx, direction="h")
                self.op.update_zoom(ratio, text="adj_z: rht")
                direction = "rht"
            _dy = foy - ioy
            if (_dy > 0):
                ratio = self.zoom_to(-_dy, direction="v")
                self.op.update_zoom(ratio, text="adj_z: btm")
                direction = "btm"
        else:
            # はみ出していない場合: do-nothing
            logger.info("[adjust_by_zoom] no adjust")
            direction = ""

        _framing_bbox = self.current_frame()
        logger.info("--- [adjust_by_zoom] frame: %s -> %s, direction: %s ---"
                    % (framing_bbox, _framing_bbox, direction))
        return direction

    def adjust_image(self):
        """
        shift/zoomによりはみ出した領域を補正
        """
        logger.info("+++ [adjust_image] start +++")
        zoom, dx, dy = None, None, None
        try:
            if (self.is_miss()):
                # TML枠内で画像の欠損がある場合（マスク切れが発生する場合）:
                # 画像拡大（TML枠縮小）により欠損領域を埋める
                _di = self.adjust_by_zoom()
        except (ZoomLimitException, ShiftLimitException) as e:
            # 許容値越え(TML枠外)例外の場合
            # 補正で例外発生しているので、画像端の欠損が生じている。あきらめシーケンス。
            # todo: 例外キーを保存
            logger.info("[adjust_image] raise OutOfOrderException. ")
            raise OutOfOrderException(
                "cannot adjust because of exception in adjust_image().")
        except (OutOfTMLException) as e:
            # TML枠内に収まらなかった場合
            # todo: 例外キーを保存
            pass
        logger.info("--- [adjust_image] done. ---")

    def adjust_for_denyobj(self):
        """
        禁止オブジェクト領域をZoom(拡大)により枠外へ追い出す。
        横長お膳画像の場合は専用トリミングとする。縦長お膳画像はトリミングしない。
        """
        logger.info("+++ [adjust_for_denyobj] start +++")
        # TODO: 禁止オブジェクトがあるか？チェック
        if (not self.deny_objs) or (not self.deny_contours):
            logger.info("[adjust_for_denyobj] No deny Objs. ")
            logger.info("--- [adjust_for_denyobj] done. ---")
            return
        # ズーム率(最大115%)にセット
        self.op.set_max_zoom(settings.max_zoom_deny)
        if ('deny_zen' in self.deny_objs):
            # お膳がある場合
            logger.info("[adjust_for_denyobj] There is deny_zen. start.")
            if self.photo.is_portrait():
                # 縦写真のときは処理しない
                logger.info(
                    "[adjust_for_denyobj] deny_zen is portrait. No trimming for deny.")
                logger.info("--- [adjust_for_denyobj] done. ---")
                return
            # お膳用トリミングルール適用
            self._trim_deny_zen()
            logger.info("[adjust_for_denyobj] There is deny_zen. end.")
        else:
            # お膳なしトリミング
            logger.info("[adjust_for_denyobj] into _trim_deny_zen. ")
            self._trim_denyobj()
        logger.info("--- [adjust_for_denyobj] done. ---")

    def _trim_denyobj(self):
        """
        禁止オブジェクトトリミングルール(お膳以外)
        """
        zoom_ratio = self.get_ratio_gradually(
            self.deny_contours, self.current_frame())
        if (zoom_ratio is None):
            logger.info("[_trim_denyobj] Zoom ratio is reached min_zoom.")
            return
        if zoom_ratio < 1.000:
            # 拡大のみ取り扱う(ここでのzoom_ratioはTML枠の拡縮率なので1未満とする)
            try:
                self.op.update_zoom(zoom_ratio, text="deny")
            except (ZoomLimitException) as e:
                # 許容値越え(TML枠外)例外の場合
                # 補正で例外発生しているので、画像端の欠損が生じている。あきらめシーケンス。
                # -> zoom許容値超えのとき:「１つ前に戻す」
                logger.info("[_trim_denyobj] raise ZoomLimitException. ")
                pass  # とりあえず通常処理
        if self.op.is_deny_in_operations():
            # 禁止オブジェクト操作が行われているとき
            try:
                # パディングつきで最小被覆領域がTML枠内か判定
                self.check_zoom_for_person_at_deny()
            except (ZoomLimitException) as e:
                # 許容値越え(TML枠外)例外の場合
                logger.info(
                    "[_trim_denyobj] raise ZoomLimitException. (check_zoom_for_person_at_deny)")
                # denyオペレーションが失敗のときは禁止Obj処理前に戻す
                self.return_back_before_deny()

    def _trim_deny_zen(self):
        """
        横写真でかつ、お膳(禁止オブジェクト)がある場合のトリミングルール
        """
        # ご破算(全トリミングご破算)にするかフラグ
        is_broken = False
        # お膳用Iライン取得準備
        self.photo.is_deny_zen = True
        # 最小被覆領域をheadでシュリンクする
        self.shrink_body_by_head()
        try:
            # お膳用Iラインに頭〜お膳までの高さが入るようにzoom
            self.zoom_to_iline_for_deny_zen()
            # お膳用Iライン下部にお膳上部をあわせる
            self.shift_to_iline_for_deny_zen()
            # TML枠(外)に対して画像が欠損していればご破算指示
            if self.is_miss():
                is_broken = True
        except (ZoomLimitException, ShiftLimitException) as e:
            # zoom, shift許容値超えの場合
            logger.info("[trim_deny_zen] raise %s (iline): %s" %
                        (type(e), str(e)))
            is_broken = True
        if is_broken is False:
            # ご破算指示がなければ終了
            return
        else:
            # 一度トリミング全てをご破算に
            self.op.put_back_init(text="pb init")
            is_broken = False
        try:
            # シフト処理のみ実施
            self.shift_to_iline_for_deny_zen_2()
            # TML枠(外)に対して画像が欠損していればご破算指示
            if self.is_miss():
                is_broken = True
        except ShiftLimitException as e:
            # shift許容値超えの場合
            logger.info("[trim_deny_zen] raise %s (iline 2nd): %s" %
                        (type(e), str(e)))
            is_broken = True
        if is_broken is True:
            # トリミング全てをご破算に
            self.op.put_back_init(text="pb init 2")
            is_broken = False

    def shrink_vbody_by_deny_zen(self, deny_vbody_minimum_bbox):
        """
        禁止判定用vbody下部をTML枠上部までシュリンクして返す。
        """
        x, y, w, h = self.current_frame()
        zy = y + h
        mx, my, mw, mh = deny_vbody_minimum_bbox
        if zy < (my+mh):
            # vbodyの下部がお膳の上部よりも下にあったら
            mh = zy - my
        return (mx, my, mw, mh)

    def check_zoom_for_person_at_deny(self):
        """
        禁止オブジェクト時のzoom_for_person.
        限定された最小被覆領域がTML枠内にパディング付きで入っているか確認する。
        """
        logger.info("+++ [check_zoom_for_person_at_deny] start +++")
        # vbody_clsよりも人と同じように扱われるクラスを限定する
        _, deny_vbody_minimum_bbox, _ = self.get_minimum_area_for_body(
            self.body_objs, self.allow_objs, allow_vbody_cls=settings.deny_vbody_cls)
        # 現在の縮尺に合わせる
        # deny_vbody_minimum_bbox = self.scale_bbox_on_center(deny_vbody_minimum_bbox, self.op.zoom)
        if (not self.photo.is_portrait()) and ('deny_zen' in self.deny_objs):
            # 横写真かつ禁止Objにお膳があるとき
            deny_vbody_minimum_bbox = self.shrink_vbody_by_deny_zen(
                deny_vbody_minimum_bbox)

        # 最小被覆領域がTML枠内に入っているか？（確認）
        self.check_deny_zoom_out(deny_vbody_minimum_bbox, text="check deny")
        # 最小被覆領域がTML枠内に入っている状態でパディングを作れるか？（確認）
        self.check_deny_padding_by_zoom(deny_vbody_minimum_bbox)
        logger.info("--- [check_zoom_for_person_at_deny] done. ---")

    def check_deny_zoom_out(self, inner_bbox, text="",
                            margin=settings.beyond_any_side_margin,
                            margins=None):
        """
        禁止Obj処理後、inner_bboxがTML枠内に入っているか？確認。
        少しでも縮小の必要があれば、ZoomLimitExceptionを上げる
        """
        logger.info("+++ [zoom_out_at_deny] start +++")
        # TML枠からはみ出しているかチェック。
        # はみ出している場合はマイナス値が入る。
        _margins = margin_top, margin_btm, _, _ = self.get_margin_around(inner_bbox,
                                                                         margin=margin, margins=margins)
        # 余白の最小値を求める
        margin_min_i = _margins.argmin()
        margin_min = _margins[margin_min_i]
        logger.info("[check_deny_zoom_out] margin_min: %d" % (margin_min))
        if (margin_min < 0):
            # sys.maxintとなった辺はマイナスにはならないので対象外となっている
            # はみ出している場合: TML枠に入る縮小率を求める
            if ((margin_min == margin_top) or (margin_min == margin_btm)):
                # 縦方向の調整
                direction = "v"
            else:
                # 横方向の調整
                direction = "h"
            ratio = self.zoom_to(-margin_min, direction=direction)
            logger.info(
                "[check_deny_zoom_out] direction: %s, ratio: %s" % (direction, ratio))
            if ratio > 1.000:
                # raise ZoomLimitException("[check_deny_zoom_out]: ZoomLimitException: %s" % ratio)
                self.op.add_zoom_exception(ratio, "ch deny zo")
            # zoom処理保存
            self.op.update_zoom(ratio, text="%s: %s" %
                                (text, self.pos2direction(margin_min_i)))
        else:
            logger.info("[check_deny_zoom_out] no zoom.")

        logger.info("[check_deny_zoom_out] current_frame: %s" %
                    (self.current_frame().__str__()))
        logger.info("--- [check_deny_zoom_out] done. ---")

    def check_deny_padding_by_zoom(self, deny_vbody_minimum_bbox):
        """
        禁止Obj処理後、人(vbody)が入るようにTML枠を縮小して余白を作成する必要があるか？確認
        少しでも縮小の必要があれば、ZoomLimitExceptionを上げる
        """
        logger.info("+++ [check_deny_padding_by_zoom] start +++")
        # 通常のズームによるパディングの場合:
        # 画像端まで及ぶ辺はパディングしない
        margin_top, margin_btm, margin_lft, margin_rht = self.get_margin_around(
            deny_vbody_minimum_bbox)
        padding, _di = 0, ""
        if (self.photo.is_portrait()):
            if (self.is_wholebody()):
                # タテ・全身写真: left/right=3px+, bottom=20px+ 余白あける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["portrait"]["wholebody"]
                padding_btm, padding_h = 0, 0
                if (margin_btm < p_btm):
                    padding_btm = (p_btm - margin_btm)
                    logger.info(
                        "[check_deny_padding_by_zoom] padding_btm = %d" % (padding_btm))
                    _di = "btm"
                if ((margin_lft < p_lft) or (margin_rht < p_rht)):
                    padding_h = p_lft - min([margin_lft, margin_rht])
                    if (margin_lft < margin_rht):
                        _di = "lft"
                    else:
                        _di = "rht"
                    logger.info(
                        "[check_deny_padding_by_zoom] padding_h = %d" % (padding_h))
                padding = max([padding_btm, padding_h])
            else:
                # タテ・半身写真: left/right=3px+, top=20px+ 余白あける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["portrait"]["halfbody"]
                padding_top, padding_h = 0, 0
                if (margin_top < p_top):
                    padding_top = (p_top - margin_top)
                    logger.info(
                        "[check_deny_padding_by_zoom] padding_top = %d" % (padding_top))
                    _di = "top"
                if ((margin_lft < p_lft) or (margin_rht < p_rht)):
                    padding_h = p_lft - min([margin_lft, margin_rht])
                    if (margin_lft < margin_rht):
                        _di = "lft"
                    else:
                        _di = "rht"
                    logger.info(
                        "[check_deny_padding_by_zoom] padding_h = %d" % (padding_h))
                padding = max([padding_top, padding_h])
        else:
            if (self.is_wholebody()):
                # ヨコ・全身: top=17+ 余白空ける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["landscape"]["wholebody"]
                if (margin_top < p_top):
                    padding = (p_top - margin_top)
                    _di = "top"
            else:
                # ヨコ・半身: top=17+ 余白空ける
                p_top, p_btm, p_lft, p_rht = settings.body_padding["landscape"]["halfbody"]
                if (margin_top < p_top):
                    padding = (p_top - margin_top)
                    _di = "top"
        logger.info("[check_deny_padding_by_zoom] padding = %d" % (padding))

        if (padding):
            # TML枠を拡大してパディングを作る必要があるか？
            ratio = self.zoom_to(padding, direction="v")
            if ratio > 1.000:
                # raise ZoomLimitException("[check_deny_padding_by_zoom]: ZoomLimitException: %s" % ratio)
                self.op.add_zoom_exception(ratio, "ch deny zo person")
            # self.op.update_zoom(ratio, text="check deny padding: %s" % (_di))
            logger.info("[check_deny_padding_by_zoom] normal zoom padding = %d, zoom = %0.3f" % (
                padding, ratio))
        else:
            logger.info("[check_deny_padding_by_zoom] no zoom.")
        logger.info("--- [check_deny_padding_by_zoom] done. ---")

    def return_back_before_deny(self):
        """
        operationを禁止Obj処理前に戻す
        (denyメッセージのoperationをすべてexceptionに積み直し、zoom,shift量も再計算して適用する)
        """
        logger.info("[return_back_before_deny] start")
        self.op.deny_to_exceptions()
        logger.info("[return_back_before_deny] end")

    def write_trimmingposision(self, values):
        """
        .trimmingpositionファイルへのvaluesの追記。
        既存キーはそのまま。
        """
        pass

    def output_for_success(self):
        """
        正常系の出力処理
        """
        sect_name = "Trimming"                          # 正常系セクション
        values = {"Trimming": {}, "TrimmingFailure":  {}}
        values[sect_name]["Position"] = "%d, %d" % (self.op.x, self.op.y)
        values[sect_name]["Zoom"] = "%f" % (self.op.zoom)
        self.write_trimmingposision(values)

    def output_for_failure(self):
        """
        異常系の出力処理
        """
        sect_name = "TrimmingFailure"                   # 異常系セクション
        values = {"Trimming": {}, "TrimmingFailure":  {}}
        i = 0
        for i, value in enumerate(self.op.exceptions.all()):
            # スタックの底から値を出力
            values[sect_name]["Error%02d" % (i+1)] = value
        # 異常系発生時の最終トリミングデータを格納
        values[sect_name]["Position"] = "%d, %d" % (self.op.x, self.op.y)
        values[sect_name]["Zoom"] = "%f" % (self.op.zoom)
        # 少なくとも1回例外が発生した場合は正常系出力は初期値とする
        values["Trimming"]["Position"] = "%d, %d" % (
            self.op.init_x, self.op.init_y)
        values["Trimming"]["Zoom"] = "%f" % (self.op.init_zoom)
        self.write_trimmingposision(values)

    def create_framed_image(self, img, trimmed=False):
        """
        imageに対してTML枠をオーバーレイした写真を生成しファイルパスを返す。
        保存先はself.options['output']。
        trimmedが指定されている場合は、当該位置にTML枠をオーバーレイする。
        """
        if (trimmed):
            # トリミング後のTML枠をオーバーレイした画像を生成
            return self.checker.create_trimming_image(img, self.options["output"], self,
                                                      self.op.x, self.op.y, self.op.zoom)
        else:
            # 入力画像にTML枠をオーバーレイした画像を生成
            return self.checker.create_default_image(img, self.options["output"], self)

    def create_debug_image(self, img_fpath):
        """
        入力画像img_fpathに認識結果をオーバーレイした画像xxx_debug.jpgを生成して返す。
        """
        return self.checker.create_debug_image(img_fpath, self.options["output"], self)

    def get_chromakey_no(self):
        """
        Exif UserComment(37510)に内蔵されているクロマキー背景番号を取得する。
        取得できないときは空文字を返す。
        """
        chromakey_no = ""
        user_comment = self.photo.get_exif(self.photo.image_fpath, tagid=37510)
        # logger.info("[get_chromakey_no] user_comment: %s" % user_comment)
        if isinstance(user_comment, str):
            # ユーザコメント取得成功時
            try:
                chromakey_no = user_comment.split(",")[4]  # コンマ区切りで5番目を抜き出す
            except IndexError:
                # クロマキー画像でない場合はここ
                pass
        else:
            # ユーザコメント取得失敗(バイナリor辞書のとき): 空文字返却
            pass
        return chromakey_no

    def is_chromakey(self):
        """
        クロマキーかどうかを下記Exif情報から判断する。
        # 36868: DateTimeDigitized
        # 37510: UserComment
        """
        datetimedigized = self.photo.exif.get(36868, "")
        return bool("," in datetimedigized) or self.chromakey_no

    def is_skip_chromakey(self):
        """
        特定クロマキーか判定する。(settings.skip_trimming_chromakey_numbersに記載)
        リスト内のクロマキーIDの場合は処理スキップする為、SkipOfOrderException例外を送出する。
        """
        if (self.chromakey_no in settings.skip_trimming_chromakey_numbers):
            # 対象のクロマキー番号に該当する場合は、トリミング処理をスキップする
            msg = "Chromakey No (%s) is confirmed. No Trimming." % self.chromakey_no
            logger.info(
                "[is_skip_chromakey] raise SkipOfOrderException. (chromakey_no: %s) " % self.chromakey_no)
            raise SkipOfOrderException(msg)
        else:
            logger.info(
                "[is_skip_chromakey] nomal img (chromakey_no: %s) " % self.chromakey_no)

    def get_metadata(self):
        """ propagationとして後続へ渡す情報をまとめる
        """
        ans_x, ans_y, ans_zoom = self.photo.preview2original(
            self.op.x, self.op.y, self.op.zoom)
        meta = {
            "ans": {
                "x": ans_x,
                "y": ans_y,
                "zoom": ans_zoom
            },
            "op": {"operations": self.op.operations_as_metadata(),
                   "exceptions": self.op.exceptions_as_metadata(),
                   "x": self.op.x,
                   "y": self.op.y,
                   "zoom": round(self.op.zoom, 3),
                   "init_x": self.op.init_x,
                   "init_y": self.op.init_y,
                   "init_zoom": round(self.op.init_zoom, 3)
                   },
            "imginfo": self.photo.imginfo,
            "is_wholebody": self.is_wholebody(),
            "chromakey_no": self.chromakey_no,
            "minimum_body": self.body_minimum_bbox,
            "minimum_vbody": self.vbody_minimum_bbox,
            "minimum_allow": self.allow_minimum_bbox,
            "center": self.get_center(),
            "upper_minimum_body": getattr(self, "upper_body_minimum_bbox", {}),
            "upper_minimum_vbody": getattr(self, "upper_vbody_minimum_bbox", {}),
            "upper_minimum_allow": getattr(self, "upper_allow_minimum_bbox", {}),
            "upper_center": self.get_center(upper=True),
            "segment_human_bboxes": getattr(self, "segment_human_bboxes", []),
            "face_avesize": self.ave_bboxes(self.body_objs.get('face', [])),
            "algo": self.algo,
            "out_of_order": self.op.out_of_order,
            "dbg_center_type": self.dbg_center_type,
            "dbg_center_base": self.dbg_center_base,
            "n_people": self.photo.n_people(),
            "n_vbody": self.photo.n_vbody(),
            "exif": self.photo.exif,
            "preview_scale": self.photo.preview_scale(),  # preview画像のオリジナル画像に対する縮小率
            "db_allow_bbox": self.db_allow_bbox,  # ここで渡される座標はpreview基準
            "db_deny_bbox": self.db_deny_bbox,  # ここで渡される座標はpreview基準
            "db_operations": self.op.db_operations_as_metadata(),
        }
        return meta

    def get_output_values(self):
        """
        position/Zoom情報を出力する。
        <position>
         回転前のオリジナル写真に対するシフト量(x, y)を出力する。
         オリジナル写真のサイズはtrimming.settings.CameraSettingsから取得する。
         x: 右側が正の方向
         y: 下側が正の方向
        <Zoom>
         TML枠の拡縮率を返す。
        """
        x, y = self.op.output_position()
        zoom = self.op.output_zoom()
        return pos_str, zoom_str

    def output(self):
        """
        異常終了判定および出力
        """
        self.no += 1
        input_fpath = self.photo.image_fpath
        output_fpath = None
        dbg_fpath = None
        if (self.op.success()):
            # 例外が1回も発生しなかった場合:
            self.output_for_success()
        else:
            # 例外が少なくとも1回発生した場合:
            self.output_for_failure()

        if (self.options['output']):
            # ファイル出力指定されている場合
            if (self.photo.n_objs['person'] == 0):
                input_fpath = self.create_framed_image(self.photo.src)
                output_fpath = self.create_framed_image(self.photo.src)
                dbg_fpath = self.create_debug_image(input_fpath)
            else:
                input_fpath = self.create_framed_image(self.photo.src)
                output_fpath = self.create_framed_image(
                    self.photo.src, trimmed=True)
                dbg_fpath = self.create_debug_image(input_fpath)

        logger.info("[output] input: %s, output: %s, dbgimg: %s" %
                    (input_fpath, output_fpath, dbg_fpath))
        meta = deepcopy(self.get_metadata())
        return input_fpath, output_fpath, dbg_fpath, meta

    def trim_img(self, task_id, img_fpath, scene_id, imginfo):
        logger.info("+++ [trim_img] start: %s +++" %
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        try:
            return self._trim_img(task_id, img_fpath, scene_id, imginfo)
        except (OutOfOrderException) as e:
            logger.info(
                "[trim_img] OutOfOrderException is occuerred. %s" % (str(e)))
            # トリミング量を初期化
            self.op.update_init()
        except (SkipOfOrderException) as e:
            logger.info(
                "[trim_img] SkipOfOrderException is occuerred. %s" % (str(e)))
            # トリミング量を初期化(ただし、out_of_order=Falseとする)
            self.op.update_init_2()
        # 異常終了判定および出力（デバッグ出力含む）
        input_fpath, output_fpath, dbgimg, meta = self.output()
        return input_fpath, output_fpath, dbgimg, meta

    def _trim_img(self, task_id, img_fpath, scene_id, imginfo):
        """
        1枚の写真: img_fpathをトリミングする。
        出力(input_fpath, output_fpath)仕様は以下の通り。
        [通常時]
        input_fpath = トリミング前のオリジナル写真、output_fpath = None
        [デバッグ時(options["output"]指定時)]
        input_fpath = トリミング前のマスク写真、output_fpath = トリミング後のマスク写真
        """
        # 画像単位の初期化
        t1 = time.time()
        self.init_params_by_image(img_fpath, task_id, scene_id, imginfo)
        try:
            # ssd検出器によるオブジェクト検出処理
            self.detect_objects()
        except (NoDetectException) as e:
            # personが未検出の場合は何もしない（トリミング処理をスキップする）
            err = "NoDetectException is occurred in detect_objects. No Trimming: %s" % (
                str(e))
            logger.info("[_trim_img] raise OutOfOrderException. ")
            raise OutOfOrderException(err)
        t2 = time.time()
        # 特定クロマキーの場合, 処理スキップする。(その場合SkipOfOrderException例外送出)
        self.is_skip_chromakey()
        # 個別トリミングルール
        if (self.photo.scene_group() == "INSIDE_LIGHT"):
            self.trim_inside_light()
        elif (self.photo.scene_group() == "CHROMAKEY") or self.is_chromakey():
            self.trim_chromakey()
        elif ((self.photo.n_vbody_allows == 0) and self.photo.is_portrait()):
            # 単身＆縦写真&人と同じように扱わない許可Objが1つもない場合
            if ((self.photo.scene_group() == "COLOR_BG") and self.is_single_wholebody()):
                self.trim_color_bg()
            else:
                self.trim_normal()
        else:
            # 上記以外は標準トリミング
            self.trim_normal()

        # 禁止オブジェクトに対する処理
        self.adjust_for_denyobj()

        # shift/zoomによりはみ出した領域を補正
        self.adjust_image()

        t3 = time.time()

        # 異常終了判定および出力（デバッグ出力含む）
        input_fpath, output_fpath, dbgimg, meta = self.output()
        t4 = time.time()

        logger.info("[%d] [trim_img] done. (1) detection: %0.3f (sec), (2) trimming: %0.3f (sec), (3) output: %0.3f (sec)\n" % (
            self.no, t2 - t1, t3 - t2, t4 - t3))
        meta["p1"] = t2 - t1
        meta["p2"] = t3 - t2
        meta["p3"] = t4 - t3
        return input_fpath, output_fpath, dbgimg, meta

    def trim_normal(self):
        """
        標準トリミング
        トリミング量（zoom/shift）を返す。
        """
        logger.info("+++ [trim_normal] start +++")
        self.algo = "trim_normal"
        # ズーム処理によりTML枠内へ入れ、パディングを作る。
        self.zoom()
        # シフト処理により配置バランスを取る（センタリング＆上下余白比率調整）
        self.shift()

    def trim_inside_light(self):
        """
        【個別トリミング】 内照
        トリミング量（zoom/shift）を返す。
        """
        logger.info("+++ [trim_inside_light] start +++")
        self.algo = "trim_inside_light"
        # ズームだけでTML枠内に入れる。シフト処理はしない。
        self.zoom()
        logger.info("--- [trim_inside_light] done. ---")

    def _trim_chromakey(self):
        """
        【個別トリミング】 クロマキー
        トリミング量（zoom/shift）を返す。
        """
        logger.info("+++ [_trim_chromakey] start +++")
        self.zoom()
        if not (self.in_tml(self.vbody_minimum_bbox)):
            # vbody最小被覆領域がTML枠外の場合:
            # 元サイズに戻す
            self.op.update_zoom(1.0, text="not_in_tml")
            # vbody最小被覆領域をTML枠に入れる
            dx, dy = self.padding_by_shift_for_vbody()
        logger.info("--- [_trim_chromakey] done. ---")

    def trim_chromakey(self):
        """
        【個別トリミング】 クロマキー
        トリミング量（zoom/shift）を返す。
        """
        logger.info("+++ [trim_chromakey] start +++")
        self.algo = "trim_chromakey"
        try:
            self._trim_chromakey()
        except (ZoomLimitException, ShiftLimitException) as e:
            # 許容値越え(TML枠外)例外の場合
            # todo: 例外キーを保存
            pass
        except (OutOfTMLException) as e:
            # TML枠内に収まらなかった場合
            # todo: 例外キーを保存
            pass
        logger.info("--- [trim_chromakey] done. ---")

    def _trim_noshime(self):
        """
        *** 現在未使用 ***
        """
        logger.info("+++ [_trim_noshime] start +++")
        # 単身であることが保証されているので配列サイズは1とする
        noshime_bbox = self.allow_objs[settings.noshime_key][0]
        framing_bbox = self.current_frame()
        # のしめobj中心を中央へ移動するh-shift量を算出
        dx = self.move_to_center(framing_bbox, noshime_bbox)
        self.op.update_shift(dx, 0, text="noshime_ctr")
        # TML枠からnoshime_bboxの距離を算出（はみ出しはマイナス）
        top, btm, lft, rht = self.get_margin_around(noshime_bbox)
        # のしめobjがTML枠に対し左右パディングが固定になるようにTML枠の拡縮率を算出
        # ratio = settings.noshime_margin_x / float(lft)
        ratio = self.zoom_to(settings.noshime_margin_x - lft, direction="h")
        logger.info("[_trim_noshime] padding lft/rht. noshime_margin_x: %d, lft: %d, ratio: %0.3f" %
                    (settings.noshime_margin_x, lft, ratio))
        self.op.update_zoom(ratio, text="noshime")
        # のしめobjに対してtop marginが固定となるようにシフト
        top, btm, lft, rht = self.get_margin_around(noshime_bbox)
        dy = top - settings.noshime_margin_y
        logger.info("[_trim_noshime] padding top. noshime_margin_y: %d, top: %d" % (
            settings.noshime_margin_y, top))
        self.op.update_shift(0, dy, text="noshime_toppad")
        logger.info("--- [_trim_noshime] done. ---")

    def trim_noshime(self):
        """
        【個別トリミング】 タテ＆1名＆のしめobj
        トリミング量（zoom/shift）を返す。
        *** 現在未使用 ***
        """
        logger.info("+++ [trim_noshime] start +++")
        self.algo = "trim_noshime"
        try:
            self._trim_noshime()
        except (ZoomLimitException, ShiftLimitException) as e:
            # 許容値越え(TML枠外)例外の場合
            # todo: 例外キーを保存
            pass
        logger.info("--- [trim_noshime] done. ---")

    def zoom_to_iline(self):
        """
        最小被覆領域の高さがIラインの高さ(679px)となるよう拡縮
        """
        mx, my, mw, mh = self.body_minimum_bbox
        (tx, ty), (bx, by) = self.photo.iline()
        iline_height = by - ty
        ratio = mh / iline_height
        logger.info("[zoom_to_iline] iline_height: %d, ratio: %0.3f" %
                    (iline_height, ratio))
        self.op.update_zoom(ratio, text="iline")

    def shift_to_iline(self):
        """
        最小被覆領域の下部をIラインの下に合うよう上下シフト
        """
        top, btm = self.current_iline()
        _, i_by = btm           # Iライン下辺のyを算出
        mx, my, mw, mh = self.body_minimum_bbox         # 最小被覆領域
        m_by = my + mh          # 最小被覆領域の下辺のyを算出
        dy = m_by - i_by
        logger.info(
            "[shift_to_iline] body_btm: %d, iline_btm: %d, dy: %d" % (m_by, i_by, dy))
        self.op.update_shift(0, dy, text="v")

    def zoom_to_iline_for_deny_zen(self):
        """
        最小被覆領域の上端からお膳(禁止Obj)上端までがIラインの高さ(578px)となるよう拡縮
        """
        zy = self.photo.h
        for zen_bbox in self.deny_objs.get("deny_zen"):
            _zen_bbox = self.scale_bbox_on_center(zen_bbox, self.op.zoom)
            zy = min(zy, _zen_bbox[1])
        mx, my, mw, mh = self.scale_bbox_on_center(
            self.body_minimum_bbox, self.op.zoom)
        (tx, ty), (bx, by) = self.photo.iline()
        logger.info("[zoom_to_iline_for_deny_zen] (tx, ty)(bx, by)=(%s, %s)(%s, %s)" % (
            tx, ty, bx, by))
        iline_height = by - ty
        hz_height = zy - my
        ratio = hz_height / iline_height
        logger.info("[zoom_to_iline_for_deny_zen] iline_height: %d, head-zen_height: %s, ratio: %0.3f" %
                    (iline_height, hz_height, ratio))
        self.op.update_zoom(ratio, text="deny_zen iline")

    def shift_to_iline_for_deny_zen(self):
        """
        お膳(禁止Obj)の上端がIラインの下に合うよう上下シフト
        """
        top, btm = self.current_iline()
        _, i_by = btm           # Iライン下辺のyを算出
        zy = self.photo.h
        for zen_bbox in self.deny_objs.get("deny_zen"):
            _zen_bbox = self.scale_bbox_on_center(zen_bbox, self.op.zoom)
            zy = min(zy, _zen_bbox[1])  # お膳領域の上辺のyを算出
        dy = zy - i_by
        logger.info(
            "[shift_to_iline_for_deny_zen] zen_top: %d, iline_btm: %d, dy: %d" % (zy, i_by, dy))
        self.op.update_shift(0, dy, text="deny_zen v")

    def shift_to_iline_for_deny_zen_2(self):
        """
        お膳(禁止Obj)の上端がIラインの下に合うよう上下シフト
        ただしお膳がTML枠より下にあった場合は動かさない。
        """
        top, btm = self.current_iline()
        _, i_by = btm           # Iライン下辺のyを算出
        zy = self.photo.h
        for zen_bbox in self.deny_objs.get("deny_zen"):
            _zen_bbox = self.scale_bbox_on_center(zen_bbox, self.op.zoom)
            zy = min(zy, _zen_bbox[1])  # お膳領域の上辺のyを算出
        dy = zy - i_by
        if dy < 0:
            logger.info(
                "[shift_to_iline_for_deny_zen_2] zen_top: %d, iline_btm: %d, dy: %d" % (zy, i_by, dy))
            self.op.update_shift(0, dy, text="deny_zen2 v")
        else:
            logger.info(
                "[shift_to_iline_for_deny_zen_2] zen_top: %d, iline_btm: %d, dy: %d -> No Shift" % (zy, i_by, dy))

    def shrink_body_by_head(self):
        """
        headがperson(body)の領域の内側にある場合、headを上端としてシュリンクする。
        """
        self.body_minimum_bbox = self._shrink_body_by_head(
            self.body_minimum_bbox)
        self.vbody_minimum_bbox = self._shrink_body_by_head(
            self.vbody_minimum_bbox)

    def _shrink_body_by_head(self, minimum_bbox):
        """
        headがperson(body)の領域の内側にある場合、headを上端としてシュリンクする。
        """
        mx, my, mw, mh = minimum_bbox
        if not ("head" in self.body_objs):
            return minimum_bbox
        # head集合のうち最上位のyを算出
        top_y = self.top_y(self.body_objs["head"])
        if (top_y > my):
            dy = top_y - my
            new_bbox = mx, top_y, mw, mh-dy
            logger.info("[_shrink_body_by_head] shrinked height: %s -> %s"
                        % (minimum_bbox.__str__(), new_bbox.__str__()))
            return new_bbox
        else:
            return minimum_bbox

    def _trim_color_bg(self):
        # (1-3) カラーバック紙のとき移動上限を上げる # wada
        self.op.set_hshift(settings.min_color_bg_hshift,
                           settings.max_color_bg_hshift)

        try:
            # 最小被覆領域をheadでシュリンクする
            self.shrink_body_by_head()
            # 被写体中心をTML枠中央へシフト
            dx = self.move_to_the_center(self.current_frame())
            self.op.update_shift(dx, 0, text="h2")
        except (ZoomLimitException, ShiftLimitException) as e:
            # zoom許容値越えの場合: 後続処理を続行
            logger.info(
                "[_trim_color_bg] exception of move_to_the_center but continue posterior processing")

        try:
            # 最小被覆領域の高さがIラインの高さとなるよう拡縮
            self.zoom_to_iline()
            # 最小被覆領域の下部をIラインの下に合うよう上下シフト
            self.shift_to_iline()
        except (ZoomLimitException, ShiftLimitException) as e:
            # zoom許容値越えの場合: 後続処理を続行
            logger.info(
                "[_trim_color_bg] exception of shift_to_iline but continue posterior processing")

    def trim_color_bg(self):
        """
        【個別トリミング】 タテ＆1名＆全身＆カラーバック紙
        トリミング量（zoom/shift）を返す。
        """
        logger.info("+++ [trim_color_bg] start +++")
        self.algo = "trim_color_bg"
        try:
            self._trim_color_bg()
        except (ZoomLimitException, ShiftLimitException) as e:
            pass
        logger.info("--- [trim_color_bg] done. ---")

    def masq_detect_ssdobjs(self):
        """
        指定領域以外をマスク(Crop)した状態でオブジェクト判定する。
        人体、許可オブジェクト、禁止オブジェクトを取得する。
        """
        if (self.photo.n_people() >= settings.shift_horizontal_th):
            if self.photo.is_portrait():
                # タテ写真の場合:
                roi = self.gen_masq_area(
                    settings.masq_portrait_degree, settings.masq_portrait_unit)
            else:
                # ヨコ写真の場合:
                roi = self.gen_masq_area(
                    settings.masq_landscape_degree, settings.masq_landscape_unit)
        else:
            roi = None
        # {"person": [bbox11, bbox12, ...],
        #  "head": [bbox21, bbox22, ...], ...} フォーマットで検出結果を取得
        body_objs = self.predict_by_crops("body", [roi])
        allow_objs = self.predict_by_crops("allow", [roi])
        deny_objs = self.predict_by_crops("deny", [roi])
        glasses_objs = self.predict_by_crops(
            "glass", body_objs.get("face", []))

        # todo: bodyとvbody（擬人オブジェクト）をどう扱うか？
        results = []
        for lblstr, cnt in self.photo.n_objs.items():
            results.append("#%s: %d" % (lblstr, cnt))
        logger.info("[masq_detect_ssdobjs] %s" % (", ".join(results)))

        return body_objs, allow_objs, deny_objs, glasses_objs

    # def gen_masq_area(self, a, b):
    #     """ マスク状態のbboxを返却する。（バストアップ基準）
    #     """
    #     target_labels = ['head', 'face']
    #     for label in target_labels:
    #         if self.body_objs.get(label, []):
    #             bottom_obj = None
    #             for obj in self.body_objs.get(label):
    #                 if bottom_obj is None:
    #                     bottom_obj = obj
    #                 elif (bottom_obj[1] + bottom_obj[3]) < (obj[1] + obj[3]):
    #                     bottom_obj = obj
    #                 else:
    #                     pass
    #             x, y, w, h = self.vbody_minimum_bbox # 最小被覆領域のみ
    #             if label == 'head':
    #                 _h = bottom_obj[1] + bottom_obj[3]
    #             elif label == 'face':
    #                 _h = bottom_obj[1] + int(bottom_obj[3] * 1.5)
    #             else:
    #                 _h = h
    #             return (x, y, w, _h)
    #     return self._gen_masq_area(a, b)

    def gen_masq_area(self, degree, unit_str, bbox=None):
        """
        マスク状態の写真のbboxを返却する。(座標算出のみで実際のマスク処理は行わない。)
        degree: 画像下部からどれだけマスクするか
        unit_str: px or %
        bbox: マスク前bbox
        """
        if bbox is None:
            # x, y, w, h = self.photo.image_bbox() # 写真全体
            x, y, w, h = self.vbody_minimum_bbox  # 最小被覆領域のみ
        else:
            x, y, w, h = bbox
        if unit_str == "%":
            _h = int(h * (100-degree) * 0.01)  # %で指定されているとき
        elif unit_str == "px":
            _h = int(h - degree)  # pxで指定されているとき
        else:
            _h = h
        if 0 < _h < h:
            return (x, y, w, _h)
        else:
            return (x, y, w, h)

    def adjust_upper_vbody_minimum_bbox(self):
        """
        upper_vbodyの極端な縮めを抑制する: 最左右faceからそれの各px分は食い込まないようにする
        現在未使用。
        """
        # 1/3マスク画像から顔が判定されないときは何もしない
        if not self.upper_body_objs.get('face', []):
            return self.upper_vbody_minimum_bbox
        # 複数人物の左右の顔を探す
        left_face_box = None
        right_face_box = None
        for face in self.upper_body_objs['face']:
            face_box = self.bbox2box(face)
            if (left_face_box == None) or (right_face_box == None):
                left_face_box, right_face_box = face_box, face_box
            else:
                if left_face_box[0] > face_box[0]:
                    left_face_box = face_box
                if right_face_box[2] < face_box[2]:
                    right_face_box = face_box
        # それぞれの座標を算出
        l_x, l_y, l_w, l_h = self.box2bbox(left_face_box)
        r_x, r_y, r_w, r_h = self.box2bbox(right_face_box)
        r_x, r_y, r_ox, r_oy = right_face_box
        v_x, v_y, v_ox, v_oy = self.bbox2box(self.vbody_minimum_bbox)
        uv_x, uv_y, uv_ox, uv_oy = self.bbox2box(self.upper_vbody_minimum_bbox)
        # TODO: upper_vbodyの左右がface*2の領域よりも内側にあるときはupperを使用しない。
        # TODO: vbodyの左右がface*2の領域よりも内側にあるときはシフト量を0にする
        # TODO: upper_vbodyがvbody枠外であればvbodyの箇所まで縮小
        new_x = max(0, v_x, min((l_x - l_w), uv_x))
        new_ox = min(v_ox, max((r_ox + r_w), uv_ox))
        return self.box2bbox((new_x, uv_y, new_ox, uv_oy))

    def get_ratio_gradually(self, contours, framing_bbox, debug=False):
        """
        徐々に縮小率をあげることでフレーミング枠内に輪郭情報を含まないzoomの倍率を出力する
        debug=Trueとすることで、縮小後のフレーミング枠と枠外のマージン値を追加で出力する
        """
        if self.photo.is_portrait():
            direction = "v"
        else:
            direction = "h"
        MIN_MARGIN = 0  # 枠外からのマージン最小値
        MAX_MARGIN = min(self.photo.w, self.photo.h) / 2  # 枠外からのマージン最大値
        find_flag = False
        for delta in range(MIN_MARGIN, MAX_MARGIN):
            ratio = self.zoom_to(-delta, direction, is_log=False)
            _framing_bbox = self.scale_bbox_on_center(framing_bbox, ratio)
            if not self.is_intersection_contours(contours, _framing_bbox):
                find_flag = True
                break
        if find_flag is False:
            logger.info("[get_ratio_gradually] ratio = None")
            return None
        logger.info("[get_ratio_gradually] ratio = %f" % (ratio))
        if debug:
            return ratio, _framing_bbox, delta
        return ratio

    def filter_deny_objs(self):
        """ 輪郭集合から余計な輪郭を除外した輪郭集合を返す。
        """
        contours = []
        # 写真中心座標を記録しておく
        center_w, center_h = self.get_center_of_bbox(self.photo.image_bbox())
        for contour in self.deny_contours:
            # ラベルと面積は除外情報として重要なため予め出力
            label = self.get_denyobj_label(contour)
            area = self.contour_area(contour)
            cx, cy = self.get_center_of_bbox(self.contour2bbox(contour))
            # ・deny_clipで面積が50px以上の輪郭は採用する
            if (label is 'deny_clip') and (area > 50):
                contours.append(contour)
            # ・deny_floorで画像中心(y:1/4~3/4)にある輪郭は除外する（上部は寝転び考慮）
            elif (label is 'deny_floor') and (center_h/2 < cy < center_h*3/2):
                continue
            # ・ラベルがdeny_zenで画像上部にある輪郭は除外する
            elif (label is 'deny_zen') and (cy < center_h):
                continue
            # ・ラベルがdeny_zenで面積が500px未満の輪郭は除外する
            elif (label is 'deny_zen') and area < 500:
                continue
            # ・面積が150px未満の輪郭は除外する
            elif area < 150:
                continue
            else:
                # それ以外はすべて採用する
                contours.append(contour)
        n_deny_contours = len(self.deny_contours)  # フィルタ前の数
        n_deny = len(contours)
        logger.info("[filter_deny_objs] n_deny: %d ->  %d" %
                    (n_deny_contours, n_deny))
        return contours

    def get_contour_color(self, contour):
        """ 輪郭を定義づける色を取得する(RGB形式)
        """
        r, g, b = 0, 0, 0
        # self.segment_srcが定義されているか確認
        if getattr(self, "segment_src", None) is None:
            logger.warn("[get_contour_color] segment_src is not loaded.")
            return (r, g, b)  # デフォルトでは黒を返す
        # 輪郭の色(r,g,b)を見つける(黒以外)
        for i in range(len(contour)):
            x, y = contour[i][0]
            b, g, r = self.segment_src[y, x]
            if not (b == g == r == 0):
                break
        return (r, g, b)

    def get_denyobj_label(self, contour):
        """ 禁止オブジェクトのラベルを取得する
        """
        color = self.get_contour_color(contour)
        label = settings.denyobj_colors.get(color, "deny")
        logger.info('[get_denyobj_label] color = (%s, %s, %s)/RGB -> %s' %
                    (color[0], color[1], color[2], label))
        return label

    def get_deny_objs(self, contours):
        """ 輪郭情報から禁止オブジェクト辞書(Value:bboxes形式)を作成する(複数輪郭)
        """
        deny_objs = {}
        for contour in contours:
            deny_label, deny_bbox = self.get_deny_labeled_bbox(contour)
            if deny_label in deny_objs:
                deny_objs[deny_label].append(deny_bbox)
            else:
                deny_objs[deny_label] = [deny_bbox]
        # 何が何個取れたか？ログ出力
        results = []
        for lblstr, bboxes in deny_objs.items():
            results.append("#%s: %d" % (lblstr, len(bboxes)))
        logger.info("[get_deny_objs] %s" % (", ".join(results)))
        return deny_objs

    def get_deny_labeled_bbox(self, contour):
        """ 禁止オブジェクトのラベルと矩形領域を返却する。(単体輪郭)
        """
        deny_label = self.get_denyobj_label(contour)
        deny_bbox = self.contour2bbox(contour)
        return deny_label, deny_bbox

    def get_segment_img_fpath(self):
        """
        禁止オブジェクトのセグメンテーション画像を生成する
        APIにより実現
        """
        # PNG書き込み先のPARAMETER/trimmingpositionディレクトリを作成
        make_trimmingposition_directory(self.photo.image_fpath)
        retval = None
        headers = {'Content-Type': 'application/json'}
        data = {"input_file": self.photo.image_fpath,
                "output_dir": get_trimmingposition_dir(self.photo.image_fpath),
                "rotate": self.photo.imginfo.get("rotate", 0)
                }
        logger.info("[get_segment_img_fpath] input: %s" %
                    self.photo.image_fpath)
        try:
            res = requests.post(settings.DENY_API_ENDPOINT,
                                data=json.dumps(data), headers=headers)
            retval = res.json().get("output_file", None)
            logger.info("[get_segment_img_fpath] output: %s" % retval)
        except Exception as e:
            logger.error("[get_segment_img_fpath] Error: %s" % str(e))
        return retval

    def _get_segment_img_fpath(self):
        """ デバッグ用
        """
        _fpath, ext = os.path.splitext(self.photo.image_fpath)
        _fdir = os.path.dirname(self.photo.image_fpath)
        _fname = os.path.basename(_fpath)
        target = os.path.join(_fdir, "segment", _fname + ".png")
        is_target = os.path.exists(target)
        if not os.path.exists(target):
            target = _fpath + ".png"
        return target

    def get_human_contour_bboxes(self, masq_box=None):
        """
        人物オブジェクトのセグメンテーションを実施し矩形領域を取得する。
        （ファイルは生成しない）
        APIにより実現
        """
        retval = []
        headers = {'Content-Type': 'application/json'}
        data = {"input_file": self.photo.image_fpath,
                "rotate": self.photo.imginfo.get("rotate", 0),
                "masq_box": masq_box,
                }
        logger.info("[get_human_contour_bboxes] input: %s" %
                    self.photo.image_fpath)
        try:
            res = requests.post(settings.HUMAN_API_ENDPOINT,
                                data=json.dumps(data), headers=headers)
            _retval = res.json()
            logger.info("[get_human_contour_bboxes] resp: %s" % str(_retval))
            retval = _retval.get("bboxes", [])
            logger.info("[get_human_contour_bboxes] output: %s" % str(retval))
        except Exception as e:
            logger.error("[get_human_contour_bboxes] Error: %s" % str(e))
        return retval


def trimming_main(**options):
    """
    テスト用トリミングmain
    """
    t1 = time.time()
    trimming = Trimming(options)
    t2 = time.time()
    if not (os.path.isdir(options['image'])):
        print("%s is not a directory..." % (options['image']))
        sys.exit(1)

    for i, img_fpath in enumerate(trimming.get_files(options['image'], ext=[".jpg", ".png", ".gif"])):
        trimming.trim_img("0123456789", img_fpath, 0, {})
