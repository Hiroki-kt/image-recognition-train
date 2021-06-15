# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFilter
# from chainer import cuda
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import cv2
import os
import math
import sys
import time
import copy
# import imutils
# import munkres
# from trimming import settings
from ssdnet.voc import VOCDataset
# from logger import get_task_logger
# import pyexiv2
try:
    import piexif
except ImportError:
    piexif = None  # copy_exif()で利用


# logger = get_task_logger("trimming")


class CommonMixIn(object):
    def img_ovl(self, img_base, img_ovl, th=0):
        """
        画像img_baseとimg_ovlを比較し、th以上の差分がある部分のみimg_ovlを
        img_baseに上書きする。
        """
        # logger.info("+++ [img_ovl] start +++")
        # th以上のimg_ovlのみを残す。その他は0
        _ovl_pixels = img_ovl * (abs(img_base - img_ovl) >= th)
        _base_pixels = img_base * \
            (abs(img_base - img_ovl) < th)    # 上書きされる部分以外を残す。その他は0
        new_img = _base_pixels + _ovl_pixels
        # logger.info("--- [img_ovl] end ---")
        return new_img

    def is_cv(self, img):
        return hasattr(img, "shape")

    '''
    def is_pil(self, img):
        return hasattr(img, "size") and isinstance(img.size, tuple)
    '''

    def is_pil(self, img):
        return not self.is_cv(img)

    def cv2pil(self, img):
        """
        画像データをOpenCV -> PIL形式に変換する。
        """
        # カラー形式をBGR(OpenCV) -> RGB(PIL)に変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # PILデータ形式に変換
        img = Image.fromarray(img)
        return img

    def pil2cv(self, img):
        """
        画像データをPIL -> OpenCV形式に変換する。
        """
        # OpenCVデータ形式に変換
        img = np.asarray(img)
        # カラー形式をRGB(PIL) -> BGR(OpenCV)に変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def hwc2chw(self, img):
        """
        行列軸変換: h, w, ch (opencv) -> ch, h, w (pil)
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        return img

    def chw2hwc(self, img):
        """
        行列軸変換: ch, h, w (pil) -> h, w, ch (opencv)
        """
        img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def angle(self, x, y):
        """
        点x, yのなす角を[0:180]で返す。
        """
        dot_xy = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if (norm_x*norm_y):
            # 割り切れる場合
            cos = dot_xy / (norm_x*norm_y)
        else:
            # 分母0の場合は少なくともいずれか一方がゼロベクトルなので角度無しと判定
            return 0
        rad = np.arccos(cos)
        theta = rad * 180 / np.pi
        return int(theta)

    def zoom_bbox(self, bbox, scale):
        """
        bboxをscale倍する
        """
        x, y, w, h = bbox
        x = x * scale
        y = y * scale
        w = w * scale
        h = h * scale
        return x, y, w, h

    def zoom_bboxes(self, bboxes, scale):
        _bboxes = []
        for bbox in bboxes:
            _bbox = self.zoom_bbox(bbox, scale)
            _bboxes.append(_bbox)
        return _bboxes

    def zoom_img(self, img, scale):
        """
        imgをscale倍した画像を返す。
        """
        oh, ow = img.shape[:2]             # オリジナル画像サイズ
        new_h = int(oh * scale)
        _src = img.copy()
        new_img = self.resize(_src, height=new_h)
        return new_img

    def get_answers(self):
        """
        現在ロードしている入力画像: self.image_fnに対応するAnnotation情報を返す。
        存在しない場合は、[], []を返す。
        """
        boxes, labels = VOCDataset.get_annotations_by_image(self.image_fn)
        return boxes, labels

    def get_answers(self, img_fpath):
        """
        現在ロードしている入力画像: self.image_fnに対応するAnnotation情報を返す。
        存在しない場合は、[], []を返す。
        """
        boxes, labels = VOCDataset.get_annotations_by_image(
            img_fpath, valid=False)
        return boxes, labels

    def load_shape(self, files):
        """
        画像集合filesが与えられたとき、shapeを返す。
        ひとまず、最初の画像ファイルのshapeを全体の共通shapeとする。
        """
        for i, fpath in enumerate(files):
            try:
                img = self.read_image(fpath)
            except AttributeError as e:
                continue
            img_h, img_w = img.shape[0], img.shape[1]
            if (img_h and img_w):
                # 最初の画像のshapeを返す
                break
        return (img_h, img_w)

    def flatten_matrix(self, mat):
        """
        行列matを一次行列へ変換する。
        ex)
        [[1, 2, 3]
         [4, 5, 6]]   -> [[1, 2, 3, 4, 5, 6]]
        """
        vec = mat.flatten(0)            # 行列からベクトルへ変換
        return vec.reshape(1, len(vec))  # 1次行列の作成

    def l2norm_normalize(self, img):
        """
        L2ノルムで正規化する。
        imgをベクトルだと考えて、L2ノルムが1となるよう正規化する。
        """
        return img / np.linalg.norm(img)

    def std_normalize(self, img):
        """
        画像の正規化
        行列要素の平均0、分散1となるように正規化する。
        """
        l_mean = img.mean()
        l_var = img.var()
        img = (img - l_mean) / float(math.sqrt(l_var))
        img = img + 1.0         # 負数を排除
        return img

    def local_contrast_norm(self, img):
        """
        Local Contrast  Normalization (単一画像の正規化)
        imgを正規化し、[0:255]の画像に復元する。
        つまり、コントラストを正規化する。
        """
        img, l_mean, l_var = self.std_normalize(img)
        l_std = math.sqrt(l_var)
        # 正規化後、元画像のレンジに戻す
        return self.scale_norm(img, l_mean, l_std, max_v=255)

    def scale_norm(self, img, mean, std, max_v=255):
        """
        [0:1]に正規化するための平均, 標準偏差を受け取り、
        [0:max_v]のレンジに復元する。
        """
        img = ((img - mean) / std) * max_v
        return img

    def global_contrast_norm(self, images, scale=1.0, min_divisor=1e-8):
        """
        Global Contrast Normalization
        平均を減算しL2ノルムで長さ1となるように画像集合imagesを正規化する。
        """
        imgs = None
        for img in images:
            fimg = self.flatten_matrix(img)
            if (imgs is None):
                imgs = fimg
            else:
                imgs = np.vstack([imgs, fimg])
        # X = np.vstack(imgs)   # 行列を縦方向に連結
        X = imgs
        # Y軸ごとに平均算出し転置(np.newaxisは縦ベクトル化)
        X = X - X.mean(axis=1)[:, np.newaxis]
        # 分散の分子に該当する部分をscaleで割る
        normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
        # min_divisor未満のものは1.0に初期化
        normalizers[normalizers < min_divisor] = 1.0

        X = X / normalizers[:, np.newaxis]

        return X

    def global_std_norm(self, images):
        """
        全画像に対する平均、標準偏差を求める。
        """
        mean_sum, std_sum = 0, 0
        for i, img in enumerate(images):
            mean_sum = img.mean()
        # 全体平均の算出（平均の平均と全データの合算値の平均は同じ）
        g_mean = mean_sum / i

        for i, img in enumerate(images):
            print("[GC_Norm STD] read img: %s" % (img_fn))
            M = img - g_mean    # 各ピクセルにおいて平均との差をとる
            M = M * M           # それらを二乗する
            std_sum = M.sum()
        # 全体分散
        g_var = std_sum / (self.shape[0] * self.shape[1])
        # 全体標準偏差
        g_std = math.sqrt(var)
        return g_mean, g_std

    def norm_path(self, pathes):
        """
        ディレクトリパスを正規化する（両端のスラッシュを除去する）
        """
        retvals = []
        if (isinstance(pathes, str)):
            pathes = [pathes]
        for path in pathes:
            path = path.strip()
            if (path[0] == '/'):
                path = path[1:]
            if (path[-1] == '/'):
                path = path[:-1]
            retvals.append(path)
        return retvals

    def get_nx_library(self):
        """
        GPU_MODEに応じたnxライブラリを取得し返す
        """
        if (self.gpu_mode):
            nx = cuda.cupy
        else:
            nx = np
        return nx

    def to_cpu(self, mat):
        """
        convert data from GPU to CPU memory when GPU_MODE.
        If GPU_MODE=False, do nothing.
        """
        if (self.gpu_mode):
            return cuda.to_cpu(mat)
        else:
            return mat

    def to_gpu(self, mat):
        """
        convert data from CPU to GPU memory when GPU_MODE.
        If GPU_MODE=False, do nothing.
        """
        if (self.gpu_mode):
            return cuda.to_gpu(mat)
        else:
            return mat

    def crop(self, img, x, y, width, height):
        """
        位置(x, y)からwidth, heightの範囲の画像を切り取って返す。
        imgに対してx, y, width, heightの矩形領域がはみ出している場合は、
        width, heightを減算し取り得る最大面積を返す。
        """
        img_h, img_w = img.shape[0], img.shape[1]
        ox = x + width
        oy = y + height
        if (x + width > img_w):
            ox = img_w  # 右端
        if (y + height > img_h):
            oy = img_h  # 下端

        return img[y:oy, x:ox]

    def cut_rect(self, img, x, y, width, height):
        """
        位置(x, y)からwidth, heightの範囲の画像を切り取って返す。
        imgに対してx, y, width, heightの矩形領域がはみ出している場合は、
        元imgの最大領域を返す。
        """
        img_h, img_w = img.shape[0], img.shape[1]
        ox = x + width
        oy = y + height
        if (x + width > img_w):
            ox = img_w
        if (y + height > img_h):
            oy = img_h

        return img[y:oy, x:ox]

    def cut_rect_with_padding(self, frame, box, value=[0, 0, 0]):
        """
        frameから矩形領域boxの範囲画像を切り取って返す。
        はみ出した領域は重心を中心として不足領域を色valueでパディングする。
        """
        x, y, w, h = box
        img_h, img_w = frame.shape[0], frame.shape[1]
        # はみ出している場合はboxよりも小さな画像が返る
        # print "frame.shape:", frame.shape
        boximg = self.cut_rect(frame, x, y, w, h)
        w_th, h_th = int(round(w/2.0)), int(round(h/2.0))
        if ((boximg.shape[0] < h_th) or (boximg.shape[1] < w_th)):
            # boxのheight, widthの半分未満は切り出し対象としない
            return None
        return self.padding_rect(boximg, w, h, value=value)

    def _centroid(self, shape):
        h, w = shape
        return int(round(h/2.0)), int(round(w/2.0))

    def centroid(self, bbox):
        """
        矩形領域boxの重心を返す
        """
        x, y, w, h = bbox
        cx = x + (w / 2)
        cy = y + (h / 2)
        return int(cx), int(cy)

    def get_centroid(self, bboxes):
        """
        複数のbboxesの重心の重心を求める
        """
        if (len(bboxes) == 0):
            return None, None
        sum_x, sum_y = 0, 0
        for bbox in bboxes:
            logger.debug("[get_centroid] bbox: %s" % (bbox.__str__()))
            cx, cy = self.centroid(bbox)
            sum_x += cx
            sum_y += cy
            logger.debug("[get_centroid] sum_x: %d, sum_y: %d" %
                         (sum_x, sum_y))
        ave_x = int(sum_x / len(bboxes))
        ave_y = int(sum_y / len(bboxes))
        return ave_x, ave_y

    def ave_bboxes(self, bboxes):
        """
        bboxesの平均の幅、高さを算出
        """
        n_bboxes = len(bboxes)
        sum_w, sum_h = 0, 0
        for bbox in bboxes:
            x, y, w, h = bbox
            sum_w += w
            sum_h += h
        if (n_bboxes):
            ave_w = int(sum_w / n_bboxes)
            ave_h = int(sum_h / n_bboxes)
            return ave_w, ave_h
        else:
            # 分母が0の場合
            return 0, 0

    def padding_rect(self, img, width, height, value=0):
        """
        img外の領域をtop/bottom, left/right方向にそれぞれ均等に色valueで埋めた
        height x widthサイズの画像を返す。
        """
        # 空の画像new_img行列を作成し、valueでパディング
        img_h, img_w = img.shape[0], img.shape[1]
        dx = max(width - img_w, 0)
        dy = max(height - img_h, 0)
        # if ( (dx < 0) or (dy < 0) ):
        if ((dx == 0) and (dy == 0)):
            # 現画像が指定サイズと同等か大きい場合は何もしない
            return img
        # 指定サイズよりも現画像が小さい場合
        pl, pr = dx / 2, dx - (dx / 2)
        pt, pb = dy / 2, dy - (dy / 2)
        # padding: (top, bottom, left, right)の順でpixel指定する。
        # print "(top, bottom, left, right) = (%f, %f, %f, %f)" % ( pt, pb, pl, pr )
        # パディングカラーはvalue。
        new_img = cv2.copyMakeBorder(
            img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=value)
        # print "[padding_rect] img.shape:", img.shape
        # print "[padding_rect] new_img.shape:", new_img.shape
        return new_img

    def is_color(self, img):
        """
        imgがカラー画像ならTrue、そうで無い場合Falseを返す。
        """
        if (len(img.shape) == 2):
            return False
        else:
            return True

    def scale_bbox(self, bbox, scale):
        """
        bboxをscale倍した矩形領域を返す。
        【左上基準】
        """
        x, y, w, h = bbox
        rx, ry = round(x * scale), round(y * scale)
        rh, rw = round(h * scale), round(w * scale)
        return (rx, ry, rw, rh)

    def scale_bbox_on_center(self, bbox, scale):
        """
        bboxをscale倍した矩形領域を返す。
        【中央基準】
        """
        x, y, w, h = bbox
        logger.debug(
            "*** [scale_bbox_on_center] x: %d, y: %d, w: %d, h: %d" % (x, y, w, h))
        rh, rw = round(h * scale), round(w * scale)
        # 拡縮された幅・高さの半分を減算（大きくなる場合）、加算（小さくなる場合）
        dx = int((rw - w) / 2)
        dy = int((rh - h) / 2)
        logger.debug("*** [scale_bbox_on_center] dx: %d, dy: %d" % (dx, dy))
        rx = x - dx
        ry = y - dy
        logger.debug(
            "*** [scale_bbox_on_center] rx: %d, ry: %d, rw: %d, rh: %d" % (rx, ry, rw, rh))
        return (rx, ry, rw, rh)

    def resize_ratio(self, img, h):
        """
        縦横費維持で、高さを指定してリサイズする。
        """
        # reshape = self.get_shape_by_height(img.shape, h, w)
        reshape = self.get_shape_by_height(img.shape, h)
        img = self.resize(img, reshape[0], reshape[1])
        return img

    def resize(self, img, width=None, height=None, inter=cv2.INTER_AREA, aspect=True):
        if (self.is_cv(img)):
            return self._resize_cv(img, width=width, height=height, inter=inter, aspect=aspect)
        else:
            return self._resize_pil(img, width=width, height=height, inter=inter, aspect=aspect)

    def _resize_cv(self, img, width=None, height=None, inter=cv2.INTER_AREA, aspect=True):
        """
        縦横比維持してリサイズする。
        短辺側に対して指定サイズでリサイズされる。
        interでは、補完技術を指定できる。
        cv2.INTER_NEAREST   : 最近傍補間
        cv2.INTER_LINEAR    : バイリニア補間（デフォルト）【拡大向き】
        cv2.INTER_AREA      : 平均画素法【縮小向き】
        cv2.INTER_CUBIC     : 4×4 の近傍領域を利用するバイキュービック補間
        cv2.INTER_LANCZOS4  : 8×8 の近傍領域を利用する Lanczos法の補間
        """
        if (aspect):
            return imutils.resize(img, width=width, height=height, inter=inter)
        else:
            return cv2.resize(img, (width, height), interpolation=inter)

    def _resize_pil(self, img, width=None, height=None, inter=Image.ANTIALIAS, aspect=True):
        """
        縦横比維持してリサイズする。
        長辺側に対して指定サイズでリサイズされる。
        interでは、補完技術を指定できる。
        PIL.Image.ANTIALIAS
        PIL.Image.NEAREST
        PIL.Image.BILINEAR
        PIL.Image.HAMMING
        PIL.Image.BICUBIC
        PIL.Image.LANCZOS
        """
        if (aspect):
            img.thumbnail((width, height), inter)           # アスペクト比維持する
        else:
            img = img.resize((width, height), inter)        # アスペクト比維持しない

        return img

    def _resize(self, img, height, width):
        """
        縦横比維持せずリサイズ
        """
        _img = img.copy()        # 上位で参照されているオブジェクトはimg.resize()できないのでコピー
        img = cv2.resize(_img, (width, height))
        return img

    def resize_image(self, img, target_shape, pad=True, blur=False, aspect=True):
        # logger.info("+++ [resize_image] start pad: %s, blur: %s, aspect: %s +++" %
        #             (pad, blur, aspect))
        #logger.info("[resize_image] target shape: %s" % target_shape)
        #logger.info("[resize_image] orig shape: %s" % img.shape)
        if (self.is_cv(img)):
            # OpenCV
            retval = self._resize_image_cv(
                img, target_shape, pad=pad, blur=blur, aspect=aspect)
        else:
            # PIL
            retval = self._resize_image_pil(
                img, target_shape, pad=pad, blur=blur, aspect=aspect)
        # logger.info("--- [resize_image] end ---")
        return retval

    def _resize_image_cv(self, img, target_shape, pad=True, blur=False, aspect=True):
        h, w = target_shape
        ih, iw = img.shape[:2]
        if (aspect == False):
            # アスペクト比無視してリサイズ
            _img = self.resize(img, height=h, width=w, aspect=aspect)
        elif (iw < ih):
            # 縦長の場合: タテを指定サイズにする。
            _img = self.resize(img, height=h, aspect=aspect)
        else:
            # 横長の場合: ヨコを指定サイズにする。
            _img = self.resize(img, width=w, aspect=aspect)
        if (blur):
            # GaussianBlur
            _img = self.blur_image(_img, use_gaussian=True)
        if (pad):
            # 余白領域を黒でpadding（背景画像のセンターに配置）
            _img = self.padding_rect(_img, w, h, value=0)
        return _img

    def _resize_image_pil(self, img, target_shape, pad=True, blur=False, aspect=True):
        h, w = target_shape
        iw, ih = img.size
        if (aspect == False):
            # アスペクト比無視してリサイズ
            _img = self.resize(img, height=h, width=w, aspect=aspect)
            bgimg = Image.new('RGB', (w, h))
        elif (iw < ih):
            # 縦長の場合: タテを指定サイズにする。
            _img = self.resize(img, width=h, height=h, aspect=aspect)
            bgimg = Image.new('RGB', (h, h))
        else:
            # 横長の場合: ヨコを指定サイズにする。
            _img = self.resize(img, width=w, height=w, aspect=aspect)
            bgimg = Image.new('RGB', (w, w))
        if (blur):
            # GaussianBlur
            _img = self.blur_image(_img, use_gaussian=True)
        if (pad):
            # 余白領域を黒でpadding（背景画像のセンターに配置）
            _iw, _ih = _img.size
            x = int((w - _iw) / 2)
            y = int((w - _ih) / 2)
            bgimg.paste(_img, (x, y))
            _img = bgimg
        return _img

    def resize_images(self, images, target_shape, pad=True, blur=False, aspect=True):
        """
        アスペクト比維持してリサイズ。
        長辺側に対して指定サイズでリサイズされる。
        pad=True: target_shapeでリサイズした余白部分をパディングする。
        """
        h, w = target_shape
        _images = []
        for (crop_i, img) in images:
            try:
                _img = self.resize_image(
                    img, target_shape, pad=pad, blur=blur, aspect=aspect)
            except cv2.error as e:
                print("[Error] CV2: %s" % (str(e)))
            _images.append((crop_i, _img))
        return _images

    def resize_with_crop(self, img, target_shape):
        """
        画像imgをtarget_shape(h, w)で定義されたサイズにリサイズし、cropした画像を返す。
        """
        if not (self.is_grayscale(img)):
            # カラー画像の場合はグレースケールに変換
            img = self.color2gray(img)
        h, w = target_shape[0], target_shape[1]
        # reshape = self.get_shape_by_height(img.shape, h, w)
        reshape = self.get_shape_by_height(img.shape, h)
        orig_shape = img.shape
        # 画像をリサイズ
        img = self.resize(img, reshape[0], reshape[1])
        reshaped_shape = img.shape
        if (reshape[1] > w):
            # 幅が指定幅を超えた場合:
            # 高さはhで固定なので、幅がwを超えた場合、中央でcropする。
            x = (reshape[1] - w) / 2
            # target_shapeのサイズにcropする
            img = self.cut_rect(img, x, 0, w, h)
        elif ((reshape[1] == w) and (reshape[0] == h)):
            # オリジナル画像とサイズ同じ場合
            pass
        else:
            # 幅が指定幅を満たしていない場合:
            # 不足領域をゼロ（黒）で埋める。黒で埋めるとconvolusionはたたみ込んだ領域の内積を
            # とるので、影響が少ない。
            # padding_color = 0           # 黒
            padding_color = img.mean()    # 平均色でパディング
            # img = self.padding_rect(img, w, h, value=255)
            img = self.padding_rect(img, w, h, value=padding_color)

        # print "orig: %s -> %s -> %s -> img: %s" % ( orig_shape, reshape, reshaped_shape, img.shape )
        # 「--shape」で指定した期待したサイズであることを保証
        assert(list(img.shape) == list(target_shape))

        return img

    def get_shape_by_height(self, shape, new_height):
        """
        new_heightが与えられたとき、縦横比を維持したshapeを返す。
        必ず
        shape.height = new_height.height
        となる。
        """
        s_height, s_width = shape[0], shape[1]
        ratio = new_height / float(s_height)
        new_width = int(s_width * ratio)
        return (new_height, new_width)

    def equalize_hist(self, img):
        """
        [0:255]の範囲のヒストグラムの出現頻度を平坦化する。
        つまり、濃度値の画素数をすべて同じ数にするわけではなく累積度数分布が直線となるよう均一化処理を行う。
        x = [0:255]のそれぞれの度数が平坦化するのではなく、どのxに元のヒストグラムのどのYを配置すると、
        累積度数分布が直線となるか？を考える。
        なので、コントラストが高いものは低く、低いものは高くなる。
        """
        img = cv2.equalizeHist(img)
        return img

    def sigmoid_contrast(self, img, a=10):
        """
        輝度をルックアップテーブルの定義に従い一対一写像変換する。
        白をより白く、黒をより黒く変化させる。
        白・黒に近い輝度が白・黒に変化する分コントラストが向上する。
        """
        if not (hasattr(self, "__lookup")):
            self.__lookup = [
                np.uint8(255.0 / (1 + math.exp(-a * (i - 128.0) / 255.0))) for i in range(256)]
        newimg = np.array([self.__lookup[v] for v in img.flat], dtype=np.uint8)
        newimg = newimg.reshape(img.shape)
        return newimg

    def blur_image(self, img, use_gaussian=True):
        """
        画像: imgに対してぼかしエフェクトをかける。
        ぼかしはガウシアンメソッドと移動平均法を選択可能。
        ガウシアンメソッドの方が、輪郭がハッキリする。
        """
        if (self.is_cv(img)):
            return self._blur_image_cv(img, use_gaussian=use_gaussian)
        else:
            return self._blur_image_pil(img, use_gaussian=use_gaussian)

    def _blur_image_cv(self, img, use_gaussian=True):
        """
        画像: imgに対してぼかしエフェクトをかける。
        ぼかしはガウシアンメソッドと移動平均法を選択可能。
        ガウシアンメソッドの方が、輪郭がハッキリする。
        """
        ave_square = (25, 25)   # 平均化する画素の周囲の大きさを指定する。大きくするほどぼやける。
        sigma_x = 1             # X軸方向の標準偏差
        if (use_gaussian):
            # gaussian
            return cv2.GaussianBlur(img, ave_square, sigma_x)
        else:
            # 移動平均
            return cv2.blur(img, ave_square)

    def _blur_image_pil(self, img, use_gaussian=True):
        """
        画像: imgに対してぼかしエフェクトをかける。
        ぼかしはガウシアンメソッドと移動平均法を選択可能。
        ガウシアンメソッドの方が、輪郭がハッキリする。
        """
        if (use_gaussian):
            return img.filter(ImageFilter.GaussianBlur(1))
        else:
            return img.filter(ImageFilter.BoxBlur(1))

    def load_image(self, img_fpath):
        """
        JVIS形式で保存された画像をロードする。
        すべての画像は横向きで保存されているので、ローテーションが必要。
        """
        # 入力画像の行列データ(BGR形式でロード)
        img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        if (hasattr(self, "imginfo")):
            img = self.rotate_image(img, self.imginfo.get("rotate", 0))
        return img

    def _load_image(self, img_fpath):
        """
        JVIS形式で保存された画像をロードする。
        回転しないver.
        """
        # 入力画像の行列データ(BGR形式でロード)
        img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        return img

    def read_image(self, ifn, shape=None):
        """
        self.width, self.heightでリサイズした画像を行列データにして返す。
        # 縦横比は維持したまま、self.shape[0] (=height)で拡大・縮小する。
        """
        # OpenCV version
        img = cv2.imread(ifn, cv2.IMREAD_UNCHANGED)          # BGR形式でロード
        if (shape):
            # 高さself.shape[0]でサイズを揃えはみ出し・余白部分を黒でcrop
            img = self.resize_with_crop(img, shape)        # グレースケールのみ対応
        return img

    '''
    def read_image(self, ifn, resize=True):
        """
        self.width, self.heightでリサイズした画像を行列データにして返す。
        # 縦横比は維持したまま、self.shape[0] (=height)で拡大・縮小する。
        """
        # OpenCV version
        img = cv2.imread(ifn)          # BGR形式でロード
        if ( resize and self.shape ):
            # 高さself.shape[0]でサイズを揃えはみ出し・余白部分を黒でcrop
            img = self.resize_with_crop(img, self.shape)        # グレースケールのみ対応
        return img
    '''

    def color2gray(self, color_img):
        """
        カラー画像（3次元行列）をグレースケール画像（2次元画像）に変換
        """
        return cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    def gray2color(self, gray_img):
        """
        グレースケール画像（2次元画像）をカラー画像（3次元行列）に変換
        """
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    def saveimg(self, img, dir_path, fn, i=None):
        if (self.is_cv(img)):
            return self._saveimg_cv(img, dir_path, fn, i=i)
        else:
            return self._saveimg_pil(img, dir_path, fn, i=i)

    def _saveimg_cv(self, img, dir_path, fn, i=None):
        """
        fnの拡張子の画像形式に従ってimg画像をdir_path/fnへ保存する。
        """
        img_path = os.path.join(dir_path, fn)
        cv2.imwrite(img_path, img)
        if (i):
            print("[save %d] image: %s" % (i, img_path))
        else:
            print("[save] image: %s" % (img_path))
        return img_path

    def _saveimg_pil(self, img, dir_path, fn, i=None):
        """
        fnの拡張子の画像形式に従ってimg画像をdir_path/fnへ保存する。
        """
        print(type(img))
        img_path = os.path.join(dir_path, fn)
        img.save(img_path)
        if (i):
            print("[save %d] image: %s" % (i, img_path))
        else:
            print("[save] image: %s" % (img_path))
        return img_path

    def savejpg(self, img, opath, quality=100):
        """
        JPEGをQUALITY=100で保存する。
        cv2デフォルトはQUALITY=95。
        """
        cv2.imwrite(opath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print("[save] image: %s" % (opath))
        return opath

    def savejpg_pil(self, img, opath, quality=95, info={}):
        """
        PIL形式のJPEGをQUALITY=95で保存する。推奨上限は95。
        100まで指定できるが圧縮アルゴリズムが一部無効となり、かつ、
        品質が向上しないデータサイズが巨大なものとなるため非推奨。
        """
        if ("exif" in info and info["exif"]) and ("dpi" in info and info["dpi"]):
            img.save(opath, quality=quality,
                     dpi=info["dpi"], exif=info["exif"])
        elif ("exif" in info and info["exif"]):
            img.save(opath, quality=quality, exif=info["exif"])
        elif ("dpi" in info and info["dpi"]):
            img.save(opath, quality=quality, dpi=info["dpi"])
        else:
            img.save(opath, quality=quality)
        print("[save] image: %s" % (opath))
        return opath

    def binary_image(self, srcimg, th=10, maxv=255):
        """
        閾値th以下をmaxvの色に変換。それ以外は0 (black)に変換する。
        """
        if (hasattr(srcimg, "shape") and (len(srcimg.shape) == 3)):
            # カラー画像の場合
            grayimg = self.color2gray(srcimg)
        else:
            grayimg = srcimg
        th, binimg = cv2.threshold(grayimg, th, maxv, cv2.THRESH_BINARY)
        return binimg

    def _dir2cls(self, fpath):
        """
        fpathからクラスラベルを取得する。
        2つ上までのディレクトリをクラスラベルとして取得する。
        ex.) /Users/s-mita/opencv_qdai_seminor/ai/data/samples/images/7/pos01046.pngの場合、
             -> 「images/7」をクラスラベルとして取得
        """
        path, filename = os.path.split(fpath)
        upper_dirs = path.rsplit("/", 2)[-2:]
        upper_dir_label = "/".join(upper_dirs)
        return upper_dir_label

    def get_files(self, path, ext=None, bn=None, islower=True, sort=False, ignore_dir=None):
        """
        path配下の（pathを含む）全ファイルを走査し、拡張子がextにマッチするものだけを返す。
        ext=Noneの場合は全ファイルを返す。
        bnに指定がある場合は、拡張子を除いたbasenameと合致しているファイルパスだけを返す。
        ex.) ext = ".png", "png", ".txt", ["png", ".txt"]など（複数指定も可）
        """
        if (ext):
            if (isinstance(ext, str)):
                # 単一指定の場合
                ext = [ext.strip()]
            for i, e in enumerate(ext):
                e = e.strip()
                if (e[0] != "."):
                    # .始まりじゃない場合は補完
                    ext[i] = "." + e
                else:
                    ext[i] = e

        if (islower):  # デフォルトでは拡張子は小文字化
            for i, e in enumerate(ext):
                ext[i] = e.lower()

        file_list = []
        for (root, dirs, files) in os.walk(path):
            for f in files:
                fpath = os.path.join(root, f)
                prev, fext = os.path.splitext(fpath)
                basename = os.path.basename(prev)
                if (ext and (fext.lower() not in ext)):
                    # 拡張子指定で、拡張子がマッチしなかった場合
                    continue
                elif ignore_dir and (ignore_dir in fpath):
                    # パス内にignore_dirがあるものは無視
                    continue
                if not (bn):
                    file_list.append(fpath)
                elif (basename == bn):
                    # bn指定がある場合は合致したものだけを残す
                    file_list.append(fpath)
        if sort is True:
            file_list.sort()
        return file_list

    def get_images(self, dirpath, ext=None, limit=0):
        """
        path配下の（pathを含む）全ファイルを走査し、拡張子がextにマッチするものに限定して画像ロードする。
        画像データのリストを返す。
        ext=Noneの場合は全ファイルを返す。
        ex.) ext = ".png", "png", ".txt", ["png", ".txt"]など（複数指定も可）
        """
        images, files = [], []
        max_h, max_w, min_h, min_w = 0, 0, sys.maxsize, sys.maxsize
        for i, fpath in enumerate(self.get_files(dirpath, ext=ext)):
            if (limit and (i > limit)):
                print("[get_images] over limit: %d" % (limit))
                break
            # img = self.read_image(fpath, resize=False)
            img = self.read_image(fpath)
            if (img is None):
                continue
            images.append(img)
            h, w = img.shape[0], img.shape[1]
            max_h, max_w = max(h, max_h), max(w, max_w)
            min_h, min_w = min(h, min_h), min(w, min_w)
            print("[%d] read img (%dx%d): %s" % (i, h, w, fpath))
            files.append(fpath)
        print("[get_images] %d images DONE. min(h, w) = (%d, %d), max(h, w) = (%d, %d)"
              % (i+1, max_h, max_w, min_h, min_w))
        return images, files

    def get_dirs(self, path):
        """
        path配下の（pathを含む）全ディレクトリを返す。
        """
        dir_list = []
        for (root, dirs, files) in os.walk(path):
            for d in dirs:
                dir_list.append(os.path.join(root, d))
        return dir_list

    def search_file(self, dirname, fn):
        """
        dirname配下にファイルfnがあれば、ファイルパスを返す。
        複数ある場合は最初に見付かったものを返す。
        無ければNoneを返す。
        """
        for d in self.get_dirs(dirname):
            fpath = os.path.join(d, fn)
            if (os.path.isfile(fpath)):
                return fpath
        return None

    def mkdirs(self, dirpath):
        """
        dirpathが無ければ作成する。あれば何もしない。
        """
        if not (os.path.isdir(dirpath)):
            os.makedirs(dirpath)

    def showimg(self, win_lbl, img, is_wait_key=False):
        """
        ウィンドウにimgを表示する。
        is_wait_key = Trueの場合、描画の更新のためのキー入力待ちをする。
        """
        if (img is None):
            return
        cv2.imshow(win_lbl, img)
        k = cv2.waitKey(1)
        if (is_wait_key):
            ipt = input()

    def showimgs(self, win_lbl, imgs, is_wait_key=False):
        """
        複数画像を横に結合してウィンドウ表示する。
        is_wait_key = Trueの場合、描画の更新のためのキー入力待ちをする。
        """
        all_img = cv2.hconcat(imgs)
        self.showimg(win_lbl, all_img, is_wait_key=is_wait_key)

    def boxbase(self, box, scale):
        """
        boxの左上基点を1/scaleした値を返す。
        """
        x, y, w, h = box
        rx = int(round(x/float(scale)))
        ry = int(round(y/float(scale)))
        return (rx, ry, w, h)

    def display_box(self, frame, box, color=None, line_width=2, debug=False, display_lbl=None):
        x, y, w, h = box
        ox, oy = x + w, y + h
        cv2.rectangle(frame, (x, y), (ox, oy), color, line_width)
        if (debug):
            # boxにIDを表示
            font = cv2.FONT_HERSHEY_PLAIN
            font_size = 1.0
            disp_str = "%s: (%d, %d)" % (str(0), h, w)
            cv2.putText(frame, disp_str, (x, y), font, font_size, (255, 0, 0))
        elif (display_lbl):
            # box上部にラベル表示
            font = cv2.FONT_HERSHEY_PLAIN
            font_size = 1.0
            disp_str = "%s" % (display_lbl)
            cv2.putText(frame, disp_str, (x, y), font, font_size, color)
            # print "[display_box] %d, %d, (label) %s" % ( x, y, display_lbl )

    def height(self, img):
        h = img.shape[0]
        return h

    def width(self, img):
        w = img.shape[1]
        return w

    def color2transparent(self, img):
        """
        カラー画像をRGBA透過カラー形式に変換
        """
        # convert 4 changel image to use alpha channel
        alpha_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        return alpha_img

    def transparent(self, rgba_img, color=(250, 250, 250)):
        """
        colorで指定した値以上に明るいピクセルを透明にする。
        入力画像形式はRGBA形式とする。
        """
        w, h = self.width(rgba_img), self.height(rgba_img)
        # print "w, h = %d, %d" % ( w, h )
        for x in range(0, w):
            for y in range(0, h):
                px = rgba_img[y, x]
                if ((px[0] >= color[0]) and (px[1] >= color[1]) and (px[2] >= color[2])):
                    px[3] = 0
                    rgba_img[y, x] = px
        return rgba_img

    def split_image(self, img, color=(255, 255, 255), delim_w=10):
        """
        colorで指定された色が立て一列に存在した場合にデリミタとする。
        当該デリミタで画像をsplitする。
        デリミタの幅はdelim_w (px)で指定し、デリミタは領域の幅がdelim_w以上である必要がある。
        """
        imgs = []
        w = self.width(img)
        h = self.height(img)
        # print "w, h = %d, %d" % ( w, h )
        sx, _w = 0, 0
        dw = 0
        digit = 0
        is_digit = False
        for x in range(0, w):
            digit_col = False
            for y in range(0, h):
                px = img[y, x]
                if not ((px[0] == color[0]) and (px[1] == color[1]) and (px[2] == color[2])):
                    # 数字が印字されている列の場合
                    digit_col = True
                    break

            if (digit_col == False):
                # デリミタ列の場合
                dw += 1
            else:
                # 数字列の場合
                dw = 0

            if ((is_digit == False) and (digit_col == True)):
                # 列xが数字の開始列の場合
                sx = x
                # print "digit start at x = %d" % ( sx )
                is_digit = True
            elif ((is_digit == True) and (digit_col == False)):
                if (dw < delim_w):
                    # デリミタ領域の幅がdelim_w未満の場合はスキップ
                    continue
                # 列xが数字の終了列の場合
                # print "digit [%d, %d]" % ( sx, x )
                is_digit = False

                # 数字画像を保存
                digit_img = self.cut_rect(img, sx, 0, x-sx-dw, h)
                imgs.append((digit, digit_img))
                print("digit %d is splited" % (digit))
                digit += 1
                dw = 0
        # 最後の列が数字列の場合は残りを保存
        if ((is_digit == True) and (digit_col == True)):
            # 数字画像を保存
            digit_img = self.cut_rect(img, sx, 0, x-sx, h)
            imgs.append((digit, digit_img))
            print("digit %d is splited" % (digit))
            digit += 1

        return imgs

    def set_items(self, ary):
        self.items = ary
        self.original_items = copy.copy(ary)                 # deep copy
        print("[set_items] #items: %d" % (len(self.items)))

    def random_select(self):
        if (len(self.items) == 0):
            self.items = copy.copy(self.original_items)      # deep copy
        choice = np.random.choice(self.items)
        self.items.remove(choice)
        return choice

    def random_sampling(self, ary):
        return np.random.choice(ary)

    def delete_pad(self, img):
        """
        画像周辺のパディングを削除
        """
        orig_h, orig_w = img.shape[:2]
        # mask = np.argwhere(img[:, :, 3] > 128) # alphaチャンネルの条件、!= 0 や == 255に調整できる
        # alphaチャンネルの条件、!= 0 や == 255に調整できる
        mask = np.argwhere(img[:, :, 2] > 128)
        (min_y, min_x) = (max(min(mask[:, 0])-1, 0), max(min(mask[:, 1])-1, 0))
        (max_y, max_x) = (min(max(mask[:, 0])+1,
                              orig_h), min(max(mask[:, 1])+1, orig_w))
        return img[min_y:max_y, min_x:max_x]

    def rotate_vec(self, x, y, degree):
        """
        点(x, y)をdegree度[0:360]回転させた座標を計算する。
        回転方向は、時計回りを正の方向とする。
        """
        # degreeをradianに変換（回転方向は時計回り）
        radian = np.radians(degree)
        rotate_mat = np.matrix([
            [np.cos(radian), -np.sin(radian)],
            [np.sin(radian), np.cos(radian)]
        ])
        point = np.matrix([[x], [y]])
        rotated_point = rotate_mat * point
        # 座標を四捨五入して返却
        return int(round(rotated_point[0])), int(round(rotated_point[1]))

    def rotate_image(self, img, degree):
        """
        opencv3のrotateを使った単純回転。
        0, 90, 180, 270, 360のみ許す。
        degree[0:360]は時計回りを正の方向とする。
        """
        degree = degree % 360
        if (degree == 0):
            return img
        elif (degree == 90):
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif (degree == 180):
            return cv2.rotate(img, cv2.ROTATE_180)
        elif (degree == 270):
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return self.affine_rotate_image(img, degree)

    def affine_rotate_image(self, img, angle, padding=False):
        """
        画像を指定した角度だけ回転させる。
        出力行列は、入力行列サイズと同じなので縦横比が異なる画像の回転には向かない。
        angle = [0:360]
        bug: 適用すると合成時に画像が途中で切れる
        """
        if ((angle == 0) or (angle == 360)):
            # 無回転指示の場合は何もしない。
            return img
        oh, ow = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((oh/2., ow/2.), angle, 1)
        img = cv2.warpAffine(img, matrix, (oh, ow))
        if (padding):
            # パディングを残す場合はそのまま返す
            return img
        else:
            # パディングを削除する場合
            return self.delete_pad(img)

    def scale_image(self, img, scale):
        """
        縦横比維持で画像を拡大縮小させる。
        """
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (int(ow*scale), int(oh*scale)))
        return self.delete_pad(img)

    def random_color_converter(self, iimg):
        """
        iimgの色をランダムに変更する。
        同じ色は同じ別の色に変換される。
        """
        # 透明情報をバックアップ
        alpha = iimg[:, :, 3]
        # colormapを適用するため一旦グレースケールに変換
        iimg = self.color2gray(iimg)
        # BGR形式でランダムなカラーマップで色変換
        # colormap = np.random.randint(0, 13)
        # 0 (red), 1 (black), 4 (orange), 11 (black&red)
        colormap = np.random.choice([0, 1, 4, 11])
        iimg = cv2.applyColorMap(iimg, colormap)
        # アルファチャンネル追加
        iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2BGRA)
        # アルファチャンネル（透明情報）を復元
        iimg[:, :, 3] = alpha
        return iimg

    def overlay(self, bg_image, overlay_image, pos_x, pos_y):
        """
        bg_imageの背景画像に対して、overlay_imageのalpha画像を貼り付ける。
        pos_xとpos_yは貼り付け時の左上の座標。
        """
        # オーバレイ画像のサイズを取得
        ol_height, ol_width = overlay_image.shape[:2]

        # OpenCVの画像データをPILに変換
        bg_image_RGB = cv2.cvtColor(
            bg_image, cv2.COLOR_BGR2RGB)     # こちらはRGBとする。
        # BGRAからRGBAへ変換
        overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

        #　PILに変換
        bg_image_PIL = Image.fromarray(bg_image_RGB)
        overlay_image_PIL = Image.fromarray(overlay_image_RGBA)

        # 合成のため、RGBAモードに変更
        bg_image_PIL = bg_image_PIL.convert('RGBA')
        overlay_image_PIL = overlay_image_PIL.convert('RGBA')

        # 同じ大きさの透過キャンパスを用意
        tmp = Image.new('RGBA', bg_image_PIL.size, (255, 255, 255, 0))
        # 用意したキャンパスに上書き
        tmp.paste(overlay_image_PIL, (pos_x, pos_y), overlay_image_PIL)
        # オリジナルとキャンパスを合成して保存
        result = Image.alpha_composite(bg_image_PIL, tmp)
        return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGR)

    def overlay_alpha(self, bg_image, overlay_image, pos_x, pos_y, ovr_alpha=0.1):
        """
        bg_imageの背景画像に対して、overlay_imageのalpha画像を貼り付ける。
        pos_xとpos_yは貼り付け時の左上の座標。
        """
        # オーバレイ画像のサイズを取得
        ol_height, ol_width = overlay_image.shape[:2]

        # OpenCVの画像データをPILに変換
        bg_image_RGB = cv2.cvtColor(
            bg_image, cv2.COLOR_BGR2RGB)     # こちらはRGBとする。
        # BGRAからRGBAへ変換
        overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

        #　PILに変換
        bg_image_PIL = Image.fromarray(bg_image_RGB)
        overlay_image_PIL = Image.fromarray(overlay_image_RGBA)

        # 合成のため、RGBAモードに変更
        bg_image_PIL = bg_image_PIL.convert('RGBA')
        overlay_image_PIL = overlay_image_PIL.convert('RGBA')

        # 同じ大きさの透過キャンパスを用意
        tmp = Image.new('RGBA', bg_image_PIL.size, (255, 255, 255, 0))
        # 用意したキャンパスに上書き
        tmp.paste(overlay_image_PIL, (pos_x, pos_y), overlay_image_PIL)

        bg_cv2 = np.asarray(bg_image_PIL)
        tmp_cv2 = np.asarray(tmp)
        cv2.addWeighted(tmp_cv2, ovr_alpha, bg_cv2, 1-ovr_alpha, 0, bg_cv2)
        # 背景画像の元フォーマット: BGRに戻す
        return cv2.cvtColor(np.asarray(bg_cv2), cv2.COLOR_RGBA2BGR)

    def random_overlay(self, bg, item, occlusion=False, alpha=True):
        """
        背景画像bgに対してアイテム画像itemをランダムな位置にオーバーレイ合成した画像を返す。
        occlusion=Trueの場合は、最大でitem全体の半分が背景画像から見切れる可能性がある。
        Falseの場合、全体がオーバーレイされる位置が選択される。
        """
        bg_h, bg_w = bg.shape[:2]
        it_h, it_w = item.shape[:2]
        if not (occlusion):
            # 見切れなしの場合
            min_x = 0
            min_y = 0
            max_x = bg_w - it_w
            max_y = bg_h - it_h
        else:
            # 見切れありの場合（幅・高さともに最大で半分が見切れる範囲を算出）
            min_x = -it_w/2
            min_y = -it_h/2
            max_x = bg_w - it_w/2
            max_y = bg_h - it_h/2
        x = np.random.randint(min_x, max_x)
        y = np.random.randint(min_y, max_y)

        if (alpha):
            # 透過データ作成許可の場合
            if (np.random.choice([False, False, True])):
                # 透過オーバーレイ
                img = self.overlay_alpha(bg, item, x, y, ovr_alpha=0.1)
            else:
                # オーバーレイ
                img = self.overlay(bg, item, x, y)
        else:
            # そのままオーバーレイする場合
            img = self.overlay(bg, item, x, y)

        # ground truth bounding boxを求める
        bx = np.maximum(x, 0)           # x, yは負値の可能性があるため最小で0とする。
        by = np.maximum(y, 0)
        # ox, oyは幅・高さを超える可能性があるため最大でbg_w, bg_hとする
        ox = np.minimum(x+it_w, bg_w)
        oy = np.minimum(y+it_h, bg_h)
        # bbox <-> (x, y, w, h) format
        bbox = (bx, by, ox-bx, oy-by)
        return img, bbox

    def area(self, bbox):
        """
        boxの面積を返す。
        """
        x, y, w, h = bbox
        return float(w * h)

    def union_box_by_point(self, box, p1):
        """
        矩形領域boxが点p1を覆う最小の矩形領域をbox形式で返す。
        """
        x, y, ox, oy = box
        _x, _y = p1
        x, y = min(x, _x), min(y, _y)
        ox, oy = max(ox, _x), max(oy, _y)
        return (x, y, ox, oy)

    def points2box(self, p1, p2):
        """
        p1, p2の2点を覆う矩形領域をbox形式で返す。
        """
        x1, y1 = p1
        x2, y2 = p2
        x, y = min(x1, x2), min(y1, y2)
        ox, oy = max(x1, x2), max(y1, y2)
        return (x, y, ox, oy)

    def union_box(self, bbox_a, bbox_b):
        """
        bbox_a, bbox_bのunion領域を覆う最小のbboxを返す。
        """
        box_a = a_x, a_y, a_ox, a_oy = self.bbox2box(bbox_a)
        box_b = b_x, b_y, b_ox, b_oy = self.bbox2box(bbox_b)
        x = min(a_x, b_x)
        y = min(a_y, b_y)
        ox = max(a_ox, b_ox)
        oy = max(a_oy, b_oy)
        box = (x, y, ox, oy)
        return box

    def top_y(self, bboxes):
        """
        bbox集合の最も上のyを返す。
        """
        min_y = sys.maxsize
        for bbox in bboxes:
            x, y, w, h = bbox
            if (min_y > y):
                min_y = y
        return min_y

    def btm_y(self, bboxes):
        """
        bbox集合の最も下のyを返す。
        """
        max_y = 0
        for bbox in bboxes:
            x, y, w, h = bbox
            if (max_y < y):
                min_y = y
        return max_y

    def lft_x(self, bboxes):
        """
        bbox集合の最も左のxを返す。
        """
        min_x = sys.maxsize
        for bbox in bboxes:
            x, y, w, h = bbox
            if (min_x > x):
                min_x = y
        return min_x

    def rht_x(self, bboxes):
        """
        bbox集合の最も右のxを返す。
        """
        max_x = 0
        for bbox in bboxes:
            x, y, w, h = bbox
            if (max_x < x):
                max_x = y
        return max_x

    def union_bbox(self, bbox_a, bbox_b):
        """
        bbox_a, bbox_bのunion領域を覆う最小のbboxを返す。
        """
        return self.box2bbox(self.union_box(bbox_a, bbox_b))

    def union(self, bbox_a, bbox_b):
        """
        bbox_a, bbox_bのunion領域の面積を返す。
        """
        intersection_area = self.intersection(bbox_a, bbox_b)
        area_a = self.area(bbox_a)
        area_b = self.area(bbox_b)
        # 2つの面積の合計から2重カウントしている領域を引く
        return area_a + area_b - intersection_area

    def intersection_bbox(self, bbox_a, bbox_b):
        """
        bbox_aとbbox_bのインターセクション領域を囲う最小のbboxを返す。
        """
        x, y, ox, oy = self.bbox2box(bbox_a)
        n_x, n_y, n_ox, n_oy = self.bbox2box(bbox_b)
        # 重なり領域算出
        start_x = max(x, n_x)
        start_y = max(y, n_y)
        end_x = min(ox, n_ox)
        end_y = min(oy, n_oy)
        return start_x, start_y, end_x-start_x, end_y-start_y

    def intersection(self, bbox_a, bbox_b):
        """
        bbox_a, bbox_bの重なり領域の面積を返す。
        """
        int_bbox = self.intersection_bbox(bbox_a, bbox_b)
        start_x, start_y, end_x, end_y = self.bbox2box(int_bbox)
        # 面積算出
        ret_w = end_x - start_x
        ret_h = end_y - start_y
        if ((ret_w > 0) and (ret_h > 0)):
            # 重なっている
            area = ret_w * ret_h
        else:
            # 重なっていない
            area = 0
        return area

    def is_point_in_box(self, point, bbox):
        """ pointがbox内部にあるかどうかをT/Fで返す
        """
        x, y = point
        x1, y1, x2, y2 = self.bbox2box(bbox)
        if (x1 <= x <= x2) and (y1 <= y <= y2):
            return True
        else:
            return False

    def max_overlaped_ratio(self, bbox, bboxes):
        """
        bboxに対してbboxとオーバーラップした領域の割合が最大のものの割合と添字番号を返す。
        bboxes中にオーバーラップしているオブジェクトが１つも見付からない場合は(0.0, None)を返す。
        """
        max_overlaped, max_i = 0.0, None
        for i, _bbox in enumerate(bboxes):
            area = self.intersection(bbox, _bbox)
            overlaped_ratio = area / self.area(bbox)
            if (overlaped_ratio > max_overlaped):
                max_overlaped = overlaped_ratio
                max_i = i
        return max_overlaped, max_i

    def bboxes_by_cls(self, bboxes, classes):
        bboxes_by_cls = {}
        for bbox, cla in zip(bboxes, classes):
            if not (cla in bboxes_by_cls):
                bboxes_by_cls[cla] = [bbox]
            else:
                bboxes_by_cls[cla].append(bbox)
        return bboxes_by_cls

    def predicts2answers(self, answers, a_boxes, predicts, p_boxes, th=0.5):
        """
        ある1つの画像に対するクラス毎の予測評価をPrecision/Recallで算出できるように、
        クラス別で以下をカウント。結果はself.accuracyに反映される。
        - n_preds: 予測数（正解・不正解関係なく）
        - n_hits: 正解数
        - n_trues: 真値の数
        - sum: 正解時のオーバーラップの総和
        <正解(HIT)の定義>
          予測側を分母とし、IoU >= thであるような真値のうち最大の真値とマッチしたという。
          その真値は、次の予測評価においてanswersから除外される。
        """
        p_bboxes = self.boxes2bboxes(p_boxes)
        a_bboxes = self.boxes2bboxes(a_boxes)
        ans_by_cls = self.bboxes_by_cls(a_bboxes, answers)
        pred_by_cls = self.bboxes_by_cls(p_bboxes, predicts)
        ans_classes = set(ans_by_cls.keys())
        pred_classes = set(pred_by_cls.keys())
        classes = ans_classes.union(pred_classes)
        # 予測・真値に登場した全クラスについて評価
        for cla in classes:
            # 予測クラスcla毎に精度評価
            cla_a_bboxes, cla_p_bboxes = [], []
            if (cla in ans_by_cls):
                cla_a_bboxes = ans_by_cls[cla]
            if (cla in pred_by_cls):
                cla_p_bboxes = pred_by_cls[cla]
            _ovl_ratios, _iou_ratios = 0.0, 0.0
            _cla_n_trues = len(cla_a_bboxes)    # 真値の個数
            _cla_n_preds = len(cla_p_bboxes)    # 予測数
            _cla_n_hits = 0             # HIT数 (予測数のうち、ratio >= thである個数)
            for p, p_bbox in enumerate(cla_p_bboxes):
                # クラスclaの真値のうち、p_bboxのオーバーラップ領域が最大となる真値を取得
                ratio, i = self.max_overlaped_ratio(p_bbox, cla_a_bboxes)
                if ((i != None) and (ratio >= th)):
                    # HIT
                    a_bbox = cla_a_bboxes.pop(i)     # 正解集合から除外
                    _cla_n_hits += 1
                    _ovl_ratios += ratio             # 正解時のオーバーラップ率の総和を算出
                    _iou_ratios += self.box_iou(a_bbox, p_bbox)

            # 真値と予測とHIT数を追記
            if not (cla in self.accuracy):
                self.accuracy[cla] = {"sum": 0, "sum_iou": 0,
                                      "n_preds": 0, "n_trues": 0, "n_hits": 0}
            # 正解時のオーバーラップの総和
            self.accuracy[cla]['sum'] += _ovl_ratios
            self.accuracy[cla]['sum_iou'] += _iou_ratios        # 正解時のIoUの総和
            # 予測数（正解・不正解関係なく）
            self.accuracy[cla]['n_preds'] += _cla_n_preds
            self.accuracy[cla]['n_trues'] += _cla_n_trues       # 真値の数
            self.accuracy[cla]['n_hits'] += _cla_n_hits         # 正解数

        return self.accuracy

    def show_accuracy(self, accuracy):
        """
        IoUベースの予測結果をPrecision/Recall/F value/Supportで出力する。
        output=Trueの場合はPretty PrintでConsole表示する。
        - Precision: 正答率
        - Recall: 再現率
        - F value: F値
        - Support: テストデータのクラス毎のサンプル数
        予測ラベルは正解でIoUが不正解の場合は常に存在しないクラス(#classes+1)を予測結果とする。
        answers, predictsの予測ラベルはsettings.labelsの添字番号(int型)とする。
        # F値のbeta=1.0で、recall&precisionを同等に扱う。
        """
        classes = settings.NUM_OF_CLASSES
        print("[Images] %s" % (self.image_dir))
        print("[Model] %s" % (self.options['model']))
        print("\t arch = %s" % (self.options['arch']))
        print("\t threshold = %s" % (self.options['th']))
        print("\t use gpu = %s" % (self.options['gpu']))
        for i, cla in enumerate(range(0, classes)):
            # クラス毎に精度を出力
            if (cla in accuracy):
                lblstr = VOCDataset.labels[cla]
                acc = accuracy[cla]
                print("-" * 100)
                print("[%d] Evaluate for <%s> " % (i+1, lblstr))
                # 精度算出
                precision = self.precision(acc['n_hits'], acc['n_preds'])
                recall = self.recall(acc['n_hits'], acc['n_trues'])
                fscore = self.fscore(precision, recall)
                print("\tn_trues: %d, n_preds: %d, n_hits: %d" %
                      (acc['n_trues'], acc['n_preds'], acc['n_hits']))
                print("\tPrecision: %0.3f" % (precision))
                # claの正解数のうち、何割正解したか？
                print("\tRecall: %0.3f" % (recall))
                print("\tFscore: %0.3f" % (fscore))
                if (acc['n_hits'] > 0):
                    # Overlap Ratio (AVE) and IoU Ratio (AVE)
                    print("\tOverlap ratio (AVE): %0.3f" %
                          (acc['sum'] / float(acc['n_hits'])))
                    print("\tIoU (AVE): %0.3f" %
                          (acc['sum_iou'] / float(acc['n_hits'])))
                else:
                    print("\tOverlap ratio (AVE): 0.0")
                    print("\tIoU (AVE): %0.0")
                print("\t#Valid Test Images: %d" % (self.n_answer_images))
                print()

    def recall(self, n_hits, n_trues):
        """
        正解数と真値数からrecallを計算
        recall: 真の正解数をどれくらい網羅したか？の割合い
        """
        if (n_trues > 0):
            recall = n_hits / float(n_trues)
        else:
            recall = 0.0
        return recall

    def precision(self, n_hits, n_preds):
        """
        予測数と正解数からprecisionを計算
        precision: 予測のうち、何割が正解だったか？の割合い
        """
        if (n_preds > 0):
            precision = n_hits / float(n_preds)
        else:
            precision = 0.0
        return precision

    def fscore(self, precision, recall):
        """
        F値を計算
        """
        if ((precision + recall) > 0):
            return (2 * precision * recall) / float(precision + recall)
        else:
            return 0.0

    def _accuracy_report(self, answers, a_bboxes, predicts, p_bboxes, th=0.5, output=False):
        """
        IoUベースの予測結果をPrecision/Recall/F value/Supportで出力する。
        output=Trueの場合はPretty PrintでConsole表示する。
        - Precision: 正答率
        - Recall: 再現率
        - F value: F値
        - Support: テストデータのクラス毎のサンプル数
        予測ラベルは正解でIoUが不正解の場合は常に存在しないクラス(#classes+1)を予測結果とする。
        answers, predictsの予測ラベルはsettings.labelsの添字番号(int型)とする。
        # F値のbeta=1.0で、recall&precisionを同等に扱う。
        """
        n_classes = settings.NUM_OF_CLASSES
        lbl_otherwise = n_classes + 1
        _predicts = []
        for lbl_a, box_a, lbl_p, box_p in zip(answers, a_bboxes, predicts, p_bboxes):
            judge, pred = False, None
            pred = lbl_p
            if (lbl_a == lbl_p):
                # 予測クラス正解の場合、IoUを評価
                judge = self.judge_iou(box_a, box_p, th=th)
                if not (judge):
                    pred = lbl_otherwise
            # IoUも加味した予測配列を作成（IoUの重なりが十分でない場合は、予測ラベルが間違うよう変更）
            _predicts.append(pred)

        precision, recall, fscore, support = precision_recall_fscore_support(
            answers, _predicts)
        if (output):
            # クラス別の精度マトリックスを表示
            print(classification_report(
                answers, _predicts, target_names=settings.labels))
        return precision, recall, fscore, support

    def judge_iou(self, box_a, box_b, th=0.5):
        """
        IoUベースの正解判定関数で正解(True)・不正解(False)を返す。
        - True: IoU(box_a, box_b) >= th
        - False: otherwise
        """
        iou = self.box_iou(box_a, box_b)
        return iou >= th

    def box_ovl(self, bbox_a, bbox_b):
        """
        bbox_aを基準としたときの2つの画像のオーバーラップ率
        ratio = a and b / a
        を求める。
        """
        area = self.intersection(bbox_a, bbox_b)
        overlaped_ratio = area / self.area(bbox_a)
        return overlaped_ratio

    def box_iou(self, bbox_a, bbox_b):
        """
        2つの画像のIoU(Intersection over Union)を求める。
        つまり、2つの画像のUnion領域に占めるIntersection領域の割合。
        [0:1]の浮動小数点を返す。
        """
        return self.intersection(bbox_a, bbox_b) / float(self.union(bbox_a, bbox_b))

    def max_iou(self, box, boxes):
        """
        boxとboxesの任意の矩形領域を調査し、最大IoUを返す。
        """
        maxiou = 0.0
        for box_i in boxes:
            iou = self.box_iou(box, box_i)
            if (iou > maxiou):
                maxiou = iou
        return maxiou

    def max_iou_pos(self, box, boxes):
        """
        boxとboxesの任意の矩形領域を調査し、最大IoUとその添字番号を返す。
        見つからない場合は、maxiou = 0, max_i = -1を返す。
        """
        maxiou, max_i = 0.0, -1
        for i, box_i in enumerate(boxes):
            iou = self.box_iou(box, box_i)
            if (iou > maxiou):
                maxiou = iou
                max_i = i
        return maxiou, max_i

    def random_hsv_trans(self, ovrimg, delta_hue=0.01, delta_sat_scale=0.5, delta_val_scale=0.5):
        """
        画像を読み込んで、hue, sat, val空間でランダム変換をした画像を返す。
        """
        hsv_image = cv2.cvtColor(ovrimg, cv2.COLOR_BGR2HSV).astype(np.float32)
        # hue
        hsv_image[:, :, 0] += int((np.random.rand()
                                  * delta_hue * 2 - delta_hue) * 255)
        # sat
        sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
        hsv_image[:, :, 1] *= sat_scale
        # val
        val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
        hsv_image[:, :, 2] *= val_scale

        hsv_image[hsv_image < 0] = 0
        hsv_image[hsv_image > 255] = 255
        hsv_image = hsv_image.astype(np.uint8)
        ovrimg = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return ovrimg

    def generate_samples(self, sample_paths, item_paths, n_items):
        """
        1個のサンプルに対して、n_items個のオーバーレイ済みサンプル画像とgrand truth座標を生成する。
        そのようなデータを入力サンプル画像の個数分繰り返す。
        """
        # scales = [0.3, 0.5, 0.7, 1.0, 1.2]
        scales = [0.3, 0.5, 0.7]
        angles = [0, 10, 180, 350]
        for i, spath in enumerate(sample_paths):
            bboxes_for_a_sample = []
            bgimg = self.read_image(spath)
            print("spath:", spath)
            for j in range(0, n_items):
                ipath = self.random_sampling(item_paths)
                print("\tipath:", ipath)
                iimg = self.read_image(ipath)
                # item画像をランダム変色させる
                iimg = self.random_color_converter(iimg)
                # item画像をランダム回転させる
                # iimg = self.rotate_image(iimg, np.random.choice(angles))
                # item画像をランダムスケールさせる
                iimg = self.scale_image(iimg, np.random.choice(scales))
                # iimgをbgimgへオーバーレイ合成する
                ovrimg, bbox = self.random_overlay(bgimg, iimg)

                if (self.max_iou(bbox, bboxes_for_a_sample) < 0.3):
                    # boxes内のboxのいずれにも0.3以上で被ってなければ追加
                    bboxes_for_a_sample.append(bbox)
                    # HSVランダム変換（todo: iimgのみに適用しても良い？）
                    ovrimg = self.random_hsv_trans(ovrimg)
                    yield (bbox, ovrimg)

    def get_label(self, ipath):
        """
        入力サンプルのクラスラベルID（整数値）を返す。
        アプリ毎にオーバーライドして定義すること。
        """
        raise NotImplemented

    def shift_bbox(self, point, bbox, reverse=False):
        """ bboxの位置を正の方向にpoint分ずらす。reverse=Trueのときは負の方向にずらす。
        """
        x, y, w, h = bbox
        sx, sy = point
        if reverse:
            _x = x - sx
            _y = y - sy
        else:
            _x = x + sx
            _y = y + sy
        return (_x, _y, w, h)

    def boxes2bboxes(self, boxes):
        """
        boxのリストboxesからbboxのリストに変換して返す。
        """
        bboxes = []
        for box in boxes:
            bboxes.append(self.box2bbox(box))
        return bboxes

    def bboxes2boxes(self, bboxes):
        """
        bboxのリストbboxesからboxのリストに変換して返す。
        """
        boxes = []
        for bbox in bboxes:
            boxes.append(self.bbox2box(bbox))
        return boxes

    def bbox2box(self, bbox):
        """
        bbox(x, y, w, h)形式からbox(x, y, ox, oy)形式に変換
        """
        x, y, w, h = bbox
        return x, y, x+w, y+h

    def box2bbox(self, bend):
        """
        box(x, y, ox, oy)形式からbbox(x, y, w, h)形式に変換
        """
        x, y, ox, oy = bend
        return x, y, ox-x, oy-y

    def distance(self, p1, p2):
        """
        p1, p2の距離を返す。
        """
        x1, y1 = p1
        x2, y2 = p2
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return int(dist)

    def degree(self, p1, p2):
        """
        2つのベクトルのなす角を返す。
        retval = [0:180]、正負の向きは考慮しない。
        """
        dot = np.dot(p1, p2)
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        cos = dot / (norm1 * norm2)
        radian = np.arccos(cos)
        theta = radian * 180 / np.pi
        return theta

    def has_overlay_number_class(self):
        """
        settings.labelsに[0:9]の数字クラスが登録されている場合はTrue。
        そうで無い場合はFalseを返す。
        """
        for num in range(0, 10):
            if (num in settings.labels):
                return True
        return False

    def generate_sample(self, sample_path, item_paths, n_items,
                        not_rois=[], scales=[0.3, 0.4, 0.5], crop_ratio=None, x=0,
                        is_train=True):
        """
        1個のサンプルに対して、n_items個のオーバーレイ済みサンプル画像とgrand truth座標を合成する。
        bboxes, labels, ovrimgを返す。ovrimgがn_items個のアイテム画像が合成された写真。
        """
        bboxes_for_a_sample = []
        for box in not_rois:
            # 合成対象外領域を予め追加
            bboxes_for_a_sample.append(self.box2bbox(box))

        bgimg = self.read_image(sample_path)
        if (crop_ratio is not None):
            # 日付領域のみcropして切り出すことで、拡大写真を予測器に渡せるようにする
            h, w = bgimg.shape[:2]
            w_crop_size = int(w * crop_ratio)
            h_crop_size = int(h * crop_ratio)
            bgimg = self.crop(bgimg, w-w_crop_size-x, h -
                              h_crop_size, w_crop_size, h_crop_size)

        boxes, labels = [], []
        n_tries = 0
        while ((len(labels) < n_items) and (n_tries < 10)):
            ipath = self.random_select()
            label = self.get_label(ipath)
            iimg = self.read_image(ipath)
            is_crop_img = ipath.find(
                "crop_numbers") >= 0       # クロップ画像かフォント画像か判別
            if (iimg.shape[2] == 3):
                # アルファチャネル追加
                iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2BGRA)
            if not (is_crop_img):
                # item画像をランダム変色させる
                iimg = self.random_color_converter(iimg)
            # item画像をランダムスケールさせる
            iimg = self.scale_image(iimg, np.random.choice(scales))
            try:
                # iimgをbgimgへオーバーレイ合成(クロップ画像は透過オーバーレイしない)
                ovrimg, bbox = self.random_overlay(
                    bgimg, iimg, alpha=not is_crop_img)
            except ValueError as e:
                print("[Error] skip this item: %s" % (str(e)))
                continue

            if (self.max_iou(bbox, bboxes_for_a_sample) <= 1.0):      # 常に許可
                # bboxes内のboxのいずれにも被ってなければ追加
                bboxes_for_a_sample.append(bbox)
                bgimg = ovrimg
                boxes.append(self.bbox2box(bbox))
                labels.append(label)
            n_tries += 1

        # print "#tries: %d" % ( n_tries )
        if (len(labels) > 0):
            # サンプル合成に成功した場合
            # HSVランダム変換（todo: iimgのみに適用しても良い？）
            if (is_train):
                ovrimg = self.random_hsv_trans(ovrimg)
            return boxes, labels, ovrimg
        else:
            # サンプル合成に失敗した場合はオリジナルを返す。
            return None, None, bgimg

    def box_distance(self, bbox1, bbox2):
        """
        box1, box2の距離を算出
        """
        p1 = self.centroid(bbox1)
        p2 = self.centroid(bbox2)
        return self.distance(p1, p2)

    def min_dist_x(self, bbox1, bbox2):
        """
        2つの矩形領域bbox1, bbox2のx方向の最小距離を返す。
        """
        if (self.intersection_bbox(bbox1, bbox2) > 0):
            # 両者に重複領域がある場合の2辺の最短距離を負の値とする
            direction = -1
        else:
            # 両者が離れている場合の2辺の最短距離を正の値とする
            direction = 1

        x1, y1, ox1, oy1 = self.bbox2box(bbox1)
        x2, y2, ox2, oy2 = self.bbox2box(bbox2)
        sides = np.array([abs(x1 - x2), abs(x1 - ox2),
                         abs(ox2 - ox1), abs(x2 - ox1)]) * direction
        i = sides.argmin()
        return sides[i]

    def min_dist_y(self, bbox1, bbox2):
        """
        2つの矩形領域bbox1, bbox2のx方向の最小距離を返す。
        """
        if (self.intersection_bbox(bbox1, bbox2) > 0):
            # 両者に重複領域がある場合の2辺の最短距離を負の値とする
            direction = -1
        else:
            # 両者が離れている場合の2辺の最短距離を正の値とする
            direction = 1

        x1, y1, ox1, oy1 = self.bbox2box(bbox1)
        x2, y2, ox2, oy2 = self.bbox2box(bbox2)
        sides = np.array([abs(x1 - x2), abs(x1 - ox2),
                         abs(ox2 - ox1), abs(x2 - ox1)]) * direction
        i = sides.argmin()
        return sides[i]

    def point_distance_matrix(self, bboxes, ids=None, allow_pairs=None):
        """
        munkres向けコスト行列を作成する。
        bbox間の距離をコストとする。
        bboxのwidth, heightを0とすれば、2点間の最小距離の総和を求める問題となる。
        """
        n_bboxes = len(bboxes)
        dist_matrix = np.zeros((n_bboxes, n_bboxes))
        # コスト行列を作成
        for i in range(0, n_bboxes):
            for j in range(0, n_bboxes):
                if (i >= j):
                    # 対称行列化で対角線と左下が選択されないようにsys.maxint化
                    dist_matrix[i, j] = sys.maxsize
                elif (ids and allow_pairs):
                    # ids, allow_pairsが与えられている場合はその制約を満たすもののみをコスト計算対象とする
                    if (self.limb_pair_id(i, j, ids) in allow_pairs):
                        # 許可されたエッジの場合
                        # 2点i, jの距離をコストとする
                        dist_matrix[i, j] = self.box_distance(
                            bboxes[i], bboxes[j])
                    else:
                        # フィルタ対象の場合は考慮させないよう最大値を与える
                        dist_matrix[i, j] = sys.maxsize
                else:
                    # ids, allow_pairsが与えられていない場合は全ペアに対をコスト計算対象とする
                    # 2点i, jの距離をコストとする
                    dist_matrix[i, j] = self.box_distance(bboxes[i], bboxes[j])
        return dist_matrix

    def limb_pair_bboxid(self, f, t, ids):
        id_f, id_t = ids[f], ids[t]
        # 関節IDでboxidをソート
        if (id_f > id_t):
            return (t, f)
        else:
            return (f, t)

    def limb_pair_id(self, f, t, ids):
        id_f, id_t = ids[f], ids[t]
        # 関節IDでid_f, id_tをソート
        if (id_f > id_t):
            tmp = id_f
            id_f = id_t
            id_t = tmp
        return (id_f, id_t)   # id_f < id_t

    def munkres(self, dist_matrix, ids=None, allow_pairs=None):
        """
        dist_matrixの全ペアのコストを最小化するようなペアリストを返す。
        pairs: [(0, 21), (1, 22), (2, 10), ...]のような感じで、
        #pairs = max(dist_matrix_width, dist_matrix_height)。
        """
        copy_distances = np.array(dist_matrix)
        # コスト行列を最小にするペアを算出
        m = munkres.Munkres()
        pairs = m.compute(dist_matrix)
        ret_pairs = []
        if (ids and allow_pairs):
            for f, t in pairs:
                if (self.limb_pair_id(f, t, ids) in allow_pairs):
                    # 許可エッジとなるペアのみ格納
                    ret_pairs.append((f, t))
            # print "*** #pairs = %d -> #ret_pairs = %d" % ( len(pairs), len(ret_pairs) )
        else:
            ret_pairs = pairs
        return ret_pairs, copy_distances

    def get_frame_bbox(self, image, trans_x=0, trans_y=0, zoom=1.):
        """
        フレーミング枠のbbox(x, y, w, h)を返す
        アスペクト比 3 : 4 または 4 : 3 の画像に対応
        """
        # フレーミング枠のデフォルト定義
        origin_shape_w = 3264
        origin_shape_h = 4352
        origin_flame_x = 236
        origin_flame_y = 361
        origin_flame_w = 3028 - origin_flame_x
        origin_flame_h = 4263 - origin_flame_y

        if image.shape[0] < image.shape[1]:
            # 縦長の画像
            x = origin_flame_x * image.shape[1] / origin_shape_h
            y = origin_flame_y * image.shape[1] / origin_shape_h
            w = origin_flame_w * image.shape[1] / origin_shape_h
            h = origin_flame_h * image.shape[1] / origin_shape_h
        else:
            # 横長の画像
            x = origin_flame_y * image.shape[0] / origin_shape_h
            y = origin_flame_x * image.shape[0] / origin_shape_h
            w = origin_flame_h * image.shape[0] / origin_shape_h
            h = origin_flame_w * image.shape[0] / origin_shape_h

        x = int(x + trans_x)
        y = int(y + trans_y)
        w = int(w * zoom)
        h = int(h * zoom)

        return x, y, w, h

    def get_center_of_bbox(self, bbox):
        """
        bboxの重心を求める
        """
        x, y, w, h = bbox
        center_x = x + w/2
        center_y = y + h/2
        return center_x, center_y

    def copy_with_exif(self, from_img_fpath, to_img_fpath):
        """
        Exif情報とともに画像コピーを実施する
        """
        im = Image.open(from_img_fpath)
        try:
            exif = im._getexif()
        except AttributeError:
            return None  # ExifないときはNone返却
        if not piexif:
            exif_bytes = im.info["exif"]
        else:
            exif_dict = piexif.load(im.info["exif"])
            exif_bytes = piexif.dump(exif_dict)
        im.save(to_img_fpath, "jpeg", exif=exif_bytes)
        return True

    def copy_exif(self, from_img_fpath, to_img_fpath):
        """
        Exif等メタデータのコピー
        https://github.com/escaped/pyexiv2
        https://blanktar.jp/blog/2014/02/python-pyexiv2.html
        """
        src = pyexiv2.ImageMetadata(from_img_fpath)
        src.read()
        dst = pyexiv2.ImageMetadata(to_img_fpath)
        dst.read()
        src.copy(dst)
        dst.write()

    def copy_exif_by_pil(self, from_img_fpath, to_img_fpath, quality=100):
        """
        Exif情報を対象画像ファイルに追加する。
        """
        exif_bytes = self.dump_exif(from_img_fpath)
        if not exif_bytes:
            return None
        im = Image.open(to_img_fpath)
        self.savejpg_pil(im, to_img_fpath, quality=quality,
                         info={"exif": exif_bytes})
        return True

    def dump_exif(self, img_fpath):
        """ Exif情報のDUMP """
        im = Image.open(img_fpath)
        try:
            exif = im._getexif()
        except AttributeError:
            return None  # ExifないときはNone返却
        if not piexif:
            exif_bytes = im.info["exif"]  # 一応これでもいける
        else:
            exif_dict = piexif.load(im.info["exif"])
            exif_bytes = piexif.dump(exif_dict)
        return exif_bytes

    def _rotate_bbox(self, bbox, rotate):
        """
        bboxを指定の角度で回転させる。

        ＊注意
        原点(0, 0)を中心として回転させるため、座標値がマイナスになる場合がある。
        その場合は、この関数外部で回転角度により画像width, heightを足しこんでやる必要がある。
        """
        # bboxをbox形式に変換。
        x, y, ox, oy = self.bbox2box(bbox)
        # (x, y)と(ox, oy)の各２点を指定分回転させる。回転は(0, 0)にを中心に行われる
        p1 = self.rotate_vec(x, y, rotate)
        p2 = self.rotate_vec(ox, oy, rotate)
        # 回転後は、bboxの起点となる位置が変化するので起点座標を計算する。
        rotated_box = self.points2box(p1, p2)
        # 回転後の矩形領域：box形式をbbox形式に戻す。
        rotated_bbox = self.box2bbox(rotated_box)
        #logger.info('********* x_dash, y_dash: %s, %s' % (x_dash, y_dash))
        #logger.info('360 - rotate: %s' % rotate)
        # (x, y, w, h = bbox
        #logger.info('x, y, w, h: %s, %s, %s, %s' % (x, y, w, h))
        #_x, _y, _w, _h = rotated_bbox
        #logger.info('_x, _y, _w, _h: %s, %s, %s, %s' % (_x, _y, _w, _h))
        return rotated_bbox

    def rotate_bbox(self, bbox, rotate, img_width, img_height):
        """
        bboxを指定の角度で回転させる。
        位置調整のためbbox, rotate以外に(bbox出力時の)画像のwidth, heightを指定する。
        """
        # bboxの位置を単純に(0, 0)起点で回転。
        _x, _y, _w, _h = self._rotate_bbox(bbox, rotate)
        # (0, 0)を起点に回転させているので、オブジェクト位置をオリジナル画像の(0, 0)基準に戻す
        if rotate == 90:
            _x += img_width
        elif rotate == 180:
            _x += img_width
            _y += img_height
        elif rotate == 270:
            _y += img_height
        else:
            pass
        return (_x, _y, _w, _h)

    def preview2orignal_bbox(self, prev_bbox, prev_scale, rotate, img_width, img_height):
        """
        preview画像基準のbbox座標系(回転後)をorignal画像基準(回転前)に変換する
        オリジナル画像に戻したときの小数点以下は四捨五入する。

        - 入力
        prev_bbox : preview画像基準bbox座標。回転後のもの。 (x, y, w, h)
        prev_scale: preview画像のオリジナル画像に対する縮小率
        rotate    : preview画像のオリジナル画像に対する回転率
        img_width : preview画像width
        img_height: preview画像height
        - 出力
        orig_bbox : オリジナル画像基準のbbox座標。回転前のもの。
        """
        # bboxを回転
        _rotate = 360 - rotate  # 指定回転分(REAL画像への回転状態に)もどす
        rotated_bbox = self.rotate_bbox(
            prev_bbox, _rotate, img_width, img_height)  # 回転処理
        # previwe画像をオリジナルサイズに戻すための倍率を計算
        original_scale = 1 / prev_scale
        # orignal画像の倍率に戻す
        orig_bbox = [x * original_scale for x in rotated_bbox]
        # 少数点を四捨五入する
        orig_bbox = list(map(int, list(map(round, (orig_bbox)))))
        return tuple(orig_bbox)

    def get_contours(self, segment_image_fpath, rotate=0, resize_w=None, resize_h=None, save_src=False):
        """
        セグメンテーションファイルから輪郭情報を抽出する
        画像をリサイズする場合は最近傍補間のためresize_w, resize_hを指定の事
        """
        src = cv2.imread(segment_image_fpath, cv2.IMREAD_COLOR)
        # 回転
        src = self.rotate_image(src, rotate)
        if resize_w and resize_h:
            # 出力画像サイズが違う場合はリサイズ(大体同じ)
            src = self.resize_segment_image(src, resize_w, resize_h)
        if save_src:
            self.segment_src = src
        # グレースケール化
        img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # しきい値指定によるフィルタリング
        retval, dst = cv2.threshold(img_gray, 250, 255, cv2.THRESH_TOZERO_INV)
        # 白黒の反転(黒ベースなのでやらない)
        # dst = cv2.bitwise_not(dst)
        # 再度フィルタリング
        retval, dst = cv2.threshold(
            dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 輪郭を抽出 (RETR_EXTERNAL指定で外側輪郭のみ抽出、CHAIN_APPROX_SIMPLEで要点のみ抽出)
        # dst, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        dst, contours, hierarchy = cv2.findContours(
            dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def resize_segment_image(self, seg_img, w, h):
        """
        画像を最近傍補間を使ってリサイズする。主にセグメンテーション画像用。
        """
        sw, sh = seg_img.shape[:2]
        if not ((w == sw) and (h == sh)):
            # logger.info(
            #     '[resize_segment_image] org/(%s, %s), seg/(%s, %s)' % (w, h, sw, sh))
            # セグメントイメージが512x512のときは最近傍補間を使ってリサイズする
            seg_img = self.resize(seg_img, width=w, height=h,
                                  inter=cv2.INTER_NEAREST, aspect=False)
        return seg_img

    def contour_center(self, contour):
        """
        1つの輪郭の重心を取得(モーメントによる取得)
        重心が同じラベルとは限らないことには注意すること
        """
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)

    def contours_centers(self, contours):
        """
        複数の輪郭の各重心座標を取得する。返却値：重心リスト
        """
        centers = []
        for contour in contours:
            centers.append(self.contour_center(contour))
        return centers

    def contour_area(self, contour):
        """
        輪郭の面積を取得する
        """
        area = cv2.contourArea(contour)
        return area

    def contour2bbox(self, contour):
        """ 1つの輪郭の外接矩形を取得する
        """
        bbox = cv2.boundingRect(contour)
        return bbox

    def contours2bboxes(self, contours):
        """ 複数の輪郭の外接矩形をリストで返す
        """
        bboxes = []
        for contour in contours:
            bboxes.append(self.contour2bbox(contour))
        return bboxes

    def is_intersection_contour(self, contour, bbox):
        """ 1つの輪郭の各点がbbox内に含まれるかどうか判定する
        """
        for i in range(len(contour)):
            x, y = contour[i][0]
            if self.is_point_in_box((x, y), bbox):
                return True
        return False

    def is_intersection_contours(self, contours, bbox):
        """
        複数の輪郭の各点がbbox内に含まれるかどうかを判定する
        1つでもbboxに交差する輪郭があればFalseとなる
        """
        _intersections = []
        for contour in contours:
            _intersections.append(self.is_intersection_contour(contour, bbox))
        if _intersections and (True in _intersections):
            return True
        return False


class Stack(object):
    def __init__(self):
        self.stack = []

    def get(self):
        return self.stack

    def push(self, value):
        """
        スタックへvalueをpushする。
        """
        self.stack.append(value)

    def pop(self):
        """
        スタックをpopしてvalueを返す。
        スタックが空の場合はIndexErrorを返す。
        """
        return self.stack.pop()

    def all(self):
        """
        stack配列を返す
        """
        return self.stack


class BboxConverter(CommonMixIn):
    """
    bbox変換クラス。
    preview画像基準のobj座標(bbox)をREAL画像基準に変換する。
    """
    # 複数アプリケーションから呼ばれるためCommonMixInに統合。
    pass
