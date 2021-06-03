# -*- coding: utf-8 -*-
from django.conf import settings
import os

# model settings
# ssd models
models = [
    {"name": "body",
     "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/body.model"),
     "th": 0.6,
     "version": 1,              # ssdnet.ssd? or ssdnet.ssd_v2?
     "labels": ('person',       # 8976
                'head',         # 8976
                'face',         # 8873
                'nodist_allow'  # 0
                )
    },
    {"name": "allow",
     # "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/allow_v3.model"),
     # "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/allow_v4.model"),
     # "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/allow_v4_2.model"),
     "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/allow_v5.model"),
     "th": 0.3,     # v2
     "version": 2,              # ssdnet.ssd? or ssdnet.ssd_v2?
     # "labels": ("allow_basket",
     #            "allow_scales",
     #            "allow_stitch",
     #            "allow_pooh",
     #            "allow_minnie",
     #            "allow_mickey",
     #            "allow_noshime",
     #            "allow_bib",
     #            "allow_lamp",
     #            "allow_clock",
     #            "allow_miffystar",
     #            "allow_moon",
     #            "allow_miffypillow",
     #            "allow_miffybroom",
     #            "allow_turtle",
     #            "allow_miffy",
     #            "allow_nameplate",
     #            "allow_mickeycloud",
     #            "allow_halfbirthday",
     #            "allow_kinprobear",
     #            "allow_kinprorabbit",
     #            "allow_flowers",
     #            "allow_flower_wreath",
     #            "allow_kirin",
     #            "allow_bd1_cake",
     #            "allow_bd2_2",
     #            # v4
     #            "allow_usamimi",
     #            "allow_poohface",
     #            "allow_mickeyface",
     #            "allow_kabuto",
     #            "allow_hinadan",
     #            "allow_bamboo",
     #            "allow_2020birthday",
     #            )
     "labels": ("allow_basket",
                "allow_bd1_cake",
                "allow_bd2_2",
                "allow_bib",
                "allow_clock",
                "allow_flowers",
                "allow_flower_wreath",
                "allow_halfbirthday",
                "allow_kinprobear",
                "allow_kinprorabbit",
                "allow_kirin",
                "allow_lamp",
                "allow_mickey",
                "allow_mickeycloud",
                "allow_miffy",
                "allow_miffybroom",
                "allow_miffypillow",
                "allow_miffystar",
                "allow_minnie",
                "allow_moon",
                "allow_nameplate",
                "allow_noshime",
                "allow_pooh",
                "allow_scales",
                "allow_stitch",
                "allow_turtle",
                "allow_usamimi",
                "allow_poohface",
                "allow_mickeyface",
                "allow_kabuto",
                "allow_hinadan",
                "allow_bamboo",
                "allow_2020birthday",
                "allow_2021birthday", # add at v5
                "allow_2021party", # add at v5
                "allow_HBDflag", # add at v4_2
                "allow_HBDbird", # add at v4_2
                "allow_HBDwood", # add at v4_2
                )
    },
    # {"name": "allow2", # iter:85000 dataset: allow2_5th
    #  "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/allow2_v1.model"),
    #  "th": 0.3,     # v2
    #  "version": 2,              # ssdnet.ssd? or ssdnet.ssd_v2?
    #  "labels": ("allow_2021birthday",
    #             "allow_2021party",
    #             )
    # },
    #{"name": "deny",
    # "th": 0.9,
    # "path": "",
    # "version": 2,              # ssdnet.ssd? or ssdnet.ssd_v2?
    # "labels": None},
    {"name": "glass",
     # "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/glass.model"), # phase2
     # "th": 0.3,
     #"path": os.path.join(settings.PROJECT_ROOT, "models/ssd/glass.model.phase3"),
     #"th": 0.6, # phase3
     # "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/glass.model.phase4.head"),
     # "th": 0.3, # phase4 (head crop)
     "path": os.path.join(settings.PROJECT_ROOT, "models/ssd/glass.model.phase4.face"),
     "th": 0.3, # phase4 (face crop)
     "version": 2,              # ssdnet.ssd? or ssdnet.ssd_v2?
     "labels": ('glasses',
                'glass',
                # 'reflection', # phase2のmodelで利用(phase3ではコメントアウトする)
                )
    },
]

# allow_obj除外設定
exclude_allow = ["allow_noshime"]

# facial keypoint detection model
facial_keypoint_detector = os.path.join(settings.PROJECT_ROOT,
                                        "models/facial_keypoint/shape_predictor_68_face_landmarks.dat")

# v2: vbody（人間と同じように扱うオブジェクト）リスト
vbody_cls = ["allow_kirin", "allow_nameplate", "allow_mickey", "allow_minnie", "allow_stitch", "allow_pooh", "allow_bd1_cake", "allow_bamboo"]
# v3: deny_vbody（禁止オブジェクト判定時、人間とおなじように扱うオブジェクト）リスト
deny_vbody_cls = ["allow_mickey", "allow_minnie", "allow_stitch", "allow_pooh"]

# min_hshift, max_hshiftを0に設定するallowクラス
shift_0_cls = ["allow_bamboo", "allow_kabuto", "allow_hinadan"]
# min_hshift, max_hshiftを小さめに設定するallowクラス
shift_small_cls = ["allow_flowers", "allow_nameplate"]

beyond_any_side_margin = 10     # (px): このマージン以上辺から離れている場合は画像端に及んでないと判定
beyond_face_padding = 30        # (px): faceが画像端に及んでいる場合のパディング値
# 余白（[top, btm, lft, rht]）の定義
body_padding = {                # bodyに対する余白の定義。Noneはpadding指定無しを意味する。
    "portrait": {
        # タテ
        "wholebody": [None, 20, 3, 3],          # 全身
        "halfbody": [20, None, 3, 3],           # 半身
    },
    "landscape": {
        "wholebody": [17, None, None, None],    # 全身
        "halfbody": [17, None, None, None],     # 半身
    }
}
th_v_wholebody = 110            # (px): faceがこの値未満の場合に全身と判定（タテ写真用）
th_h_wholebody = 130            # (px): faceがこの値未満の場合に全身と判定（ヨコ写真用）
# トリミング量の許容値
#min_hshift = -25                # x方向のシフト下限
min_hshift = -20                # x方向のシフト下限
min_vshift = -10000             # y方向のシフト下限
#min_small_hshift = -25          # x方向のシフト下限(small設定)
min_small_hshift = -20          # x方向のシフト下限(small設定)
#max_hshift = 25                 # x方向のシフト上限
max_hshift = 20                 # x方向のシフト上限
max_vshift = 10000              # y方向のシフト上限
#max_small_hshift = 25           # x方向のシフト上限(small設定)
max_small_hshift = 20           # x方向のシフト上限(small設定)
min_zoom = 0.92                 # zoomの下限
max_zoom = 1.10                 # zoomの上限
max_zoom_deny = 1.15            # zoomの上限(禁止Objあり)
min_body_h = 400                # v-shiftするminimum_bodyの高さの最小値

min_0_hshift = 0                # x方向のシフト下限(shift_0_clsがある場合の設定)
max_0_hshift = 0                # x方向のシフト上限(shift_0_clsがある場合の設定)
min_2wholebody_hshift = -50     # x方向のシフト下限(2人&全身設定)
max_2wholebody_hshift = 50      # x方向のシフト上限(2人&全身設定)
min_2halfbody_hshift = -20      # x方向のシフト下限(2人&半身設定)
max_2halfbody_hshift = 20       # x方向のシフト上限(2人&半身設定)
min_with_horizontal = -35       # x方向のシフト下限(人数がshift_horizontal_th以上の時のmin_hshift)
max_with_horizontal = 35        # x方向のシフト上限(人数がshift_horizontal_th以上の時のmax_hshift)
min_color_bg_hshift = -50       # x方向のシフト下限(trim_color_bg()設定)
max_color_bg_hshift = 50        # x方向のシフト上限(trim_color_bg()設定)

shift_horizontal_th = 3         # 閾値以上の場合、vbody枠中心でh-shift。閾値未満の場合、顔中心。

noshime_key = "allow_noshime"   # <現在未使用>
noshime_margin_x = 20           # <現在未使用> (px): のしめobjのleft/rightマージン
noshime_margin_y = 120          # <現在未使用> (px): のしめobjのtopマージン

iou_min = 0.45              # SSDのdecode時のIoUの最小値（どれだけ重なっていたら同じ検出物とするか？）
conf_min = 0.10             # SSDのdecode時の確信度の最小値
obj_ovl_threshold = 0.7 # OBJ検出時, 当該被り率以上のオブジェクトを1つとみなす(同じラベルで比較)
obj_ovl_select = 1 # OBJ検出時, 0:被り除去を行わない, 1:面積の小さいOBJを除去, 2:面積の大きいOBJを除去

masq_threshold = shift_horizontal_th  # 何人以上の時、マスクを実行するか
# masq_portrait_degree = 33   # タテ写真で、画像下部からどれだけマスクするか
masq_portrait_degree = 40   # タテ写真で、画像下部からどれだけマスクするか
masq_portrait_unit = "%"    # タテ写真でのマスク単位。% か px の選択。
# masq_landscape_degree = 20  # ヨコ写真で、画像下部からどれだけマスクするか
masq_landscape_degree = 40  # ヨコ写真で、画像下部からどれだけマスクするか
masq_landscape_unit = "%"   # ヨコ写真でのマスク単位。% か px の選択。


# トリミング処理をしないクロマキー番号リスト(オブジェクト判定は行う)
skip_trimming_chromakey_numbers = ['61-3', '61-4', '61-7', '61-8', '71-1', '71-2', '71-3']

denyobj_colors = {
    # RGB
    #(0, 0, 0)     : "deny_background", # black
    (255, 128, 0) : "deny_zen",        # orange
    (0, 0, 255)   : "deny_backpaper",  # blue
    (0, 255, 0)   : "deny_roll",       # green
    (0, 255, 255) : "deny_floor",      # ligth blue
    (255, 0, 255) : "deny_clip",       # pink
    (128, 0, 255) : "deny_staff",      # purple
    (255, 0, 0)   : "deny_stick",      # red
    # 128のケースあり
    (0, 0, 128)   : "deny_backpaper",  # blue
    (0, 128, 0)   : "deny_roll",       # green
}

# フレーミング枠のマスク画像（デバッグ用）
frame_image_path = os.path.join(settings.PROJECT_ROOT, "aisrv_v1/static/img/4DA-2_T_MD_clear.png")

# DEBUG設定
TRIM_VOC_BASE_DIRS = {
    "deny17th": "/home/lafla/wada_trimming_check/17th/",
    "deny17th_dgr": "/home/lafla/wada_trimming_check/17th_degrade_check/",
    "check1": "/home/lafla/wada_trimming_check/check/",
    "check2": "/home/lafla/wada_trimming_check/check2/",
    "check3": "/home/lafla/wada_trimming_check/check3/",
    "check4": "/home/lafla/wada_trimming_check/check4/",
    "check5": "/home/lafla/wada_trimming_check/check5/",
    "check6": "/home/lafla/wada_trimming_check/check6/",
    "check7": "/home/lafla/wada_trimming_check/check7/",
    "deny_release": "/home/lafla/wada_trimming_check/deny_release/",
    # "VOCJvis3allow_cls3": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/img/staging/1/",
    # "VOCJvis3allow_cls3": "aisrv_v1/static/img/data/train/VOCJvis3allow_cls3/JPEGImages/",
    # "VOCJvis3face": "aisrv_v1/static/img/data/test/VOCJvis3face/JPEGImages",
    # "New Nameplate": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/new_nameplate/JPEGImages",
    # "Test Data1": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data1/",
    # "CanonEOS5DMarkIII": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/CanonEOS5DMarkIII",
    # "color_single": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/color_single",
    # "hanshin": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/hanshin",
    # "hanshin_kazoku": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/hanshin_kazoku",
    # "hanshin_naisho": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/hanshin_naisho",
    # "hanshin_naisho_utubuse": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/hanshin_naisho_utubuse",
    # "katana": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/katana",
    # "koi_nobori": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/koi_nobori",
    # "maternity": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/maternity",
    # "miffy": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/miffy",
    # "nameplate": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/nameplate",
    # "tilt": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/tilt",
    # "tokushu": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/tokushu",
    # "otherwise": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data2/otherwise",
    # "inside_light": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_inside_light",
    # "inside_light_rotate": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_inside_light_rotate",
    # "chromakey": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_chromakey",
    # "Test Data3": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/test_data3/",
    # "hair_ribon": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/hair_ribon/",
    # "irregular_images": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/irregular_images/",
    # "birthday12": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/birthday12/",
    # "nas1": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/ini_list/1.txt",
    # "nas2": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/ini_list/2.txt",
    # "staging": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/ini_list/staging.txt",
    # "tokyo": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/ini_list/tokyo.txt",
    # "t0": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/ini_list/t0.txt",
    # # "staging1": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/img/staging/1/",
    # "bug imgs": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/bug_imgs",
    # "dbg20180828": "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/img/dbg20180828",
    #"dbg20180828": "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/dbg20180828/",
}


# TRIM_VOC_BASE_DIR = os.path.join(settings.PROJECT_ROOT, "aisrv_v1/static/img/data/train/VOCJvis3allow_cls3/")
# TRIM_VOC_BASE_DIR = os.path.join(settings.PROJECT_ROOT, "aisrv_v1/static/img/data/test/VOCJvis3face/")
# TRIM_VOC_BASE_DIR = os.path.join(settings.PROJECT_ROOT, "/home/jvis/git/jvis/aisrv_v1/aisrv_v1/static/img/new_nameplate")
OUTPUT_DIR = os.path.join(settings.PROJECT_ROOT, "/home/lafla/git/jvis/aisrv_v1/aisrv_v1/static/generated/")

# Camera Settings
CameraSettings = {
    "normal": {
        "shorter_side": 3264,                           # 短編(px)
        "longer_side": 4352,                            # 長辺(px)
        "framing_bbox": (212, 361, 2840, 3630),         # タテのTML枠内側(bbox)
        "framing_bbox_outer": (100, 269, 3064, 3814),   # タテのTML枠外側(bbox)
        "iline": [(1632, 806), (1632, 3762)],           # iline
        "iline_zen": [(2176, 548), (2176, 3064)],       # お膳iline(禁止Objお膳,横写真用)
    },
    "Canon EOS 5D Mark IV": {
        "shorter_side": 4480,                           # 短編(px)
        "longer_side": 6720,                            # 長辺(px)
        "framing_bbox": (87, 560, 4288, 5600),          # タテのTML枠内側(bbox)
        "framing_bbox_outer": (87, 406, 4288, 5908),    # タテのTML枠外側(bbox)
        "iline": [(2231, 1246), (2231, 5807)],          # iline（未提供なのでデフォルト値）
        "iline_zen": [(3360, 594), (3360, 4375)],       # お膳iline(禁止Objお膳,横写真用)
    },
    "Canon EOS 5D Mark III": {
        "shorter_side": 3840,                           # 短編(px)
        "longer_side": 5760,                            # 長辺(px)
        "framing_bbox": (202, 676, 3436, 4408),         # タテのTML枠内側(bbox)
        "framing_bbox_outer": (76, 598, 3688, 4564),    # タテのTML枠外側(bbox)
        "iline": [(1916, 1505), (1916, 4785)],          # iline（未提供なのでデフォルト値）
        "iline_zen": [(2880, 614), (2880, 3646)],       # お膳iline(禁止Objお膳,横写真用)
    },
    "Canon EOS 5D Mark II": {
        "shorter_side": 3744,                           # 短編(px)
        "longer_side": 5616,                            # 長辺(px)
        "framing_bbox": (226, 669, 3292,4278),          # タテのTML枠内側(bbox)
        "framing_bbox_outer": (74, 583, 3596, 4450),    # タテのTML枠外側(bbox)
        "iline": [(1866, 1185), (1866, 4665)],          # iline（未提供なのでデフォルト値）
        "iline_zen": [(2808, 594), (2808, 3561)],       # お膳iline(禁止Objお膳,横写真用)
    },
}

# JIPSトリミング初期値設定
JIPS_INIT_X = 0
JIPS_INIT_Y = 0
JIPS_INIT_ZOOM = 1.0
JIPS_INIT_ROTATE = 0        # <現在未使用>
JIPS_INIT_FINEROTATE = 0    # <現在未使用>

DENY_API_ENDPOINT = "http://127.0.0.1:8081/deny"
HUMAN_API_ENDPOINT = "http://127.0.0.1:8081/human"
