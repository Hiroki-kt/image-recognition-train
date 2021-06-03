# -*- coding: utf-8 -*-
# from common.logger import get_task_logger
from common.common import CommonMixIn
import cv2
from PIL.ExifTags import TAGS
from PIL import Image
# from scene.constant import SCENE_LABEL, TRIMMING_MAP
# from . import settings
import time

# logger = get_task_logger("trimming")


class Photo(CommonMixIn):
    """
    写真の状態・情報を管理するパラメタクラス
    """
    # 取得対象EXIFコード（機種毎に項目がバラバラのため必要最小限に統一。Codec Error防止。）
    # ImageWidth(256), ImageLength(257), ExifImageWidth(40962), ExifImageHeight(40963)
    # Make(271), Model(272), Orientation(274), FNumber(33437), SubjectDistance(37382)
    # ShutterSpeedValue(37377), FocalLength(37386), DateTimeDigitized(36868)
    EXIF_CODE = [256, 257, 40962, 40963, 271, 272,
                 274, 33437, 37382, 37377, 36868, 37386]

    def __init__(self, img_fpath, task_id, scene_id, imginfo):
        # 上位レイヤーから取得する入力情報
        self.image_fpath = img_fpath    # 入力画像のファイルパス
        self.task_id = task_id          # DBのタスクID
        self.scene_id = scene_id        # シーンID
        self.imginfo = imginfo          # extra情報（辞書型）
        # 中間データ
        t1 = time.time()
        self.src = self.load_image(self.image_fpath)
        self.org = self.src.copy()
        self.exif = self.get_exif(self.image_fpath)
        t2 = time.time()
        # 入力画像のサイズ（ROI or ORG）
        self.h, self.w = self.src.shape[:2]
        # 入力画像のサイズ（バックアップ）
        self.oh, self.ow = self.org.shape[:2]
        self.shift_x, self.shift_y = 0, 0                       # ROIのorg座標に対するx, y
        # self.is_zoom = False    # 全身写真の場合False、半身写真の場合True
        # self.is_center = None   # 被写体が中央の場合True。そうでない場合False。
        self.n_objs = {"person": 0, "head": 0,
                       "face": 0}    # 各クラスのobj数（必須項目のみ初期化）
        self.n_allows = 0         # 許可オブジェクト数（フィルター対象は除外）
        self.n_vbody_allows = 0   # 人と同じように扱う許可オブジェクト数(settings.vbody_cls)
        self.is_deny_zen = False  # 禁止オブジェクト:お膳があったかどうか
        # self.camera, self.camera_type = self.get_camera_settings(
        #     self.exif.get("Model", "normal"))

        # logger.info("[Photo.__init__] %s image loaded: %0.2f (sec)" %
        #             (self.image_fpath, t2-t1))
        # logger.info("[Photo.__init__] scene_id: %s, scene_group: %s" % (
        #     self.scene_id, self.scene_group()))

    def scene_label(self):
        """
        この写真のシーンラベルを返す
        """
        return SCENE_LABEL.get(self.scene_id, "---")

    def scene_group(self):
        """
        この写真のシーングループを返す。
        """
        for k, v in list(TRIMMING_MAP.items()):
            if self.scene_id in v:
                return k
        return None

    def get_camera_settings(self, name):
        """
        Exifのカメラ機種名（Model）から、settings.pyの対応カメラ情報を取得する。
        TML枠を機種名により変更したり、元画像サイズを取得するため。
        """
        # logger.info("+++ [get_camera_settings] Model: %s +++" % (name))
        cs = settings.CameraSettings
        if not (name in cs):
            name = "normal"
        camera = settings.CameraSettings.get(name)  # カメラ情報の取得
        # logger.info(
        #     "--- [get_camera_settings] camera_setting: %s ---" % (name))
        return camera, name

    def get_exif(self, img_fpath, tagid=None):
        """
        exif情報をimg_fpathより取得する
        * tagid指定時は指定IDのExif情報のみ取得する
        """
        im = Image.open(img_fpath)
        try:
            exif = im._getexif()
        except AttributeError:
            return {}

        if (exif is None):
            return {}

        if tagid:  # tagid指定時
            return exif.get(tagid, "")
        # decode
        exif_table = {}
        for tagid in Photo.EXIF_CODE:
            tagstr = TAGS.get(tagid, "---")
            exif_table[tagstr] = exif.get(tagid, "")
        return exif_table

    def image_bbox(self):
        """
        画像全体を表すbboxを返す
        """
        return 0, 0, self.w, self.h

    def is_portrait(self):
        """
        縦写真の場合True、横写真の場合Falseを返す。
        """
        # todo: 上位レイヤーからの入力仕様を要確認
        if (self.h > self.w):
            return True
        else:
            return False

    def count_up(self, lblstr):
        """
        クラス: lblstrに対するオブジェクト数を更新して数を返す。
        """
        if not (lblstr in self.n_objs):
            self.n_objs[lblstr] = 0
        self.n_objs[lblstr] += 1
        return self.n_objs[lblstr]

    def n_vbody(self):
        """
        人を含めたvbodyの総数を返す。
        """
        _sum = self.n_people()
        for lblstr in settings.vbody_cls:
            n_lbl = self.n_objs.get(lblstr, 0)
            _sum += n_lbl
        # logger.info("[n_vbody] return (sum): %d" % (_sum))
        return _sum

    def n_people(self):
        return max(self.n_objs['person'], self.n_objs['head'], self.n_objs['face'])

    def is_single(self):
        """
        単身写真か否かを判定する。
        bodyが未検出でも対応できるようmaxで判定する。
        """
        return self.n_people() == 1

    def landscape(self, bbox):
        """
        90度右に回転させたbboxを返す
        """
        x, y, w, h = bbox
        return y, x, h, w

    def framing_bbox(self):
        """
        デフォルトのフレーミング枠の初期値(bbox)を返す。
        """
        original_longer_side = float(
            self.camera["longer_side"])  # オリジナル写真の長辺側の長さ(pixel)
        fbbox = self.camera["framing_bbox"]
        if (self.is_portrait()):
            preview_scale = self.h / original_longer_side
            original_framing_bbox = fbbox                       # 縦画像のオリジナルのフレーミング枠
        else:
            preview_scale = self.w / original_longer_side
            # 上下左右均等にしたフレーミング枠
            original_framing_bbox = self.landscape(
                fbbox)       # 横画像のオリジナルのフレーミング枠

        retval = [round(x * preview_scale) for x in original_framing_bbox]
        # logger.debug("[framing_bbox] default = (%d, %d, %d, %d)" %
        #              (retval[0], retval[1], retval[2], retval[3]))
        return retval

    def framing_bbox_outer(self):
        """
        デフォルトのフレーミング枠外周の初期値(bbox)を返す。
        """
        original_longer_side = float(self.camera["longer_side"])
        fbbox_outer = self.camera["framing_bbox_outer"]
        if (self.is_portrait()):
            preview_scale = self.h / original_longer_side
            original_framing_bbox = fbbox_outer                 # 縦画像のオリジナルのフレーミング枠
        else:
            preview_scale = self.w / original_longer_side
            # 上下左右均等にしたフレーミング枠
            original_framing_bbox = self.landscape(
                fbbox_outer)  # 横画像のオリジナルのフレーミング枠

        # retval = [int(x * preview_scale) for x in original_framing_bbox]
        retval = [round(x * preview_scale) for x in original_framing_bbox]
        # logger.debug("[framing_bbox_outer] default = (%d, %d, %d, %d)" %
        #              (retval[0], retval[1], retval[2], retval[3]))
        return retval

    def iline(self):
        """
        デフォルトIラインの重心を上下それぞれで返す。
        iline = [(top_x, top_y), (btm_x, btm_y)]
        """
        original_longer_side = float(
            self.camera["longer_side"])    # オリジナル写真の長辺側の長さ(pixel)
        if self.is_deny_zen is True:
            _iline = self.camera["iline_zen"]
            preview_scale = self.w / original_longer_side
            original_iline = _iline
        else:
            _iline = self.camera["iline"]
            if (self.is_portrait()):
                preview_scale = self.h / original_longer_side
                original_iline = _iline  # 縦画像のオリジナルのフレーミング枠
            else:
                preview_scale = self.w / original_longer_side
                # 横画像のオリジナルのフレーミング枠
                original_iline = [(_iline[0][1], _iline[0][0]),
                                  (_iline[1][1], _iline[1][0])]

        # return [(int(x * preview_scale), int(y * preview_scale)) for x, y in original_iline]
        return [(round(x * preview_scale), round(y * preview_scale)) for x, y in original_iline]

    def preview_scale(self):
        """
        preview画像のオリジナル画像に対する縮小率を返す。
        """
        # オリジナル写真の長辺側の長さ(pixel)
        original_longer_side = float(self.camera["longer_side"])
        if (self.is_portrait()):
            preview_scale = self.h / original_longer_side
        else:
            preview_scale = self.w / original_longer_side
        return preview_scale

    def preview2original(self, x, y, zoom):
        """
        preview基準（回転済み写真）のトリミング量を
        original基準（回転前写真）に変換する。
        """
        # previwe画像をオリジナルサイズに戻すための倍率を計算
        original_scale = 1 / self.preview_scale()
        # シフト量をオリジナルスケールに変換 & 画像に対するシフト量なので極性を反転
        orig_x = x * original_scale * -1
        orig_y = y * original_scale * -1
        # 回転前の座標系に変換
        rotate = 360 - self.imginfo.get("rotate", 0)   # オリジナル画像に戻すためのdegreeを算出
        orig_x, orig_y = self.rotate_vec(orig_x, orig_y, rotate)
        # オリジナル写真に対するTML枠の拡縮率は、Preview基準と同じなためそのまま。
        # 画像に対する拡縮率として返すので逆数に変換する。
        orig_zoom = 1 / zoom
        # logger.info("[preview2original] (x, y, zoom) = preview: (%d, %d, %0.3f) -> orig: (%d, %d, %0.3f)"
        #             % (x, y, zoom, orig_x, orig_y, orig_zoom))
        return orig_x, orig_y, orig_zoom

    def org_coordinate(self, bbox):
        """
        roi座標からオリジナル座標へ変換
        """
        x, y, w, h = bbox
        x = x + self.shift_x
        y = y + self.shift_y
        return (x, y, w, h)

    def change_roi(self, bbox):
        """
        bboxで指定されたROIを処理対象とする。
        """
        x, y, w, h = bbox
        # if ( (x <= 0) or (y <= 0) ):
        if ((x < 0) or (y < 0)):
            err = "Invalid bbox error:", bbox
            raise Exception(err)
        self.src = self.cut_rect(self.src, *bbox).copy()
        self.h, self.w = self.src.shape[:2]
        # ROIのorg画像に対する開始座標を記録
        self.shift_x, self.shift_y = x, y

    def change_org(self):
        """
        処理対象を元ファイルに戻す。
        """
        self.src = self.org
        self.h, self.w = self.org.shape[:2]
        self.shift_x, self.shift_y = 0, 0

    def to_collect_bbox(self, bbox):
        """ bboxの写真外にはみ出した部分を取り除く
        """
        x, y, w, h = bbox
        if x < 0:
            w = w + x
            x = 0
        if y < 0:
            h = h + y
            y = 0
        if x + w > self.w:
            w = self.w - x
        if y + h > self.h:
            h = self.h - y
        return (x, y, w, h)
