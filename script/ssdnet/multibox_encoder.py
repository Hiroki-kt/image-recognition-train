# -*- coding: utf-8 -*-
from __future__ import division # for python2.7
import itertools
import numpy as np

from ssdnet import matrix_iou


class MultiBoxEncoder(object):

    def __init__(self, model):
        self.variance = model.variance

        default_boxes = list()
        for k in range(len(model.grids)):
            for v, u in itertools.product(range(model.grids[k]), repeat=2):
                cx = (u + 0.5) * model.steps[k]
                cy = (v + 0.5) * model.steps[k]

                s = model.sizes[k]
                default_boxes.append((cx, cy, s, s))

                s = np.sqrt(model.sizes[k] * model.sizes[k + 1])
                default_boxes.append((cx, cy, s, s))

                s = model.sizes[k]
                for ar in model.aspect_ratios[k]:
                    default_boxes.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    default_boxes.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self.default_boxes = np.array(default_boxes)

    def encode(self, boxes, labels, threshold=0.5):
        if len(boxes) == 0:
            return (
                np.zeros(self.default_boxes.shape, dtype=np.float32),
                np.zeros(self.default_boxes.shape[:1], dtype=np.int32))

        iou = matrix_iou(
            np.hstack((
                self.default_boxes[:, :2] - self.default_boxes[:, 2:] / 2,
                self.default_boxes[:, :2] + self.default_boxes[:, 2:] / 2)),
            boxes)
        gt_idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)
        boxes = boxes[gt_idx]
        labels = labels[gt_idx]

        loc = np.hstack((
            ((boxes[:, :2] + boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            np.log((boxes[:, 2:] - boxes[:, :2]) / self.default_boxes[:, 2:]) /
            self.variance[1]))

        conf = 1 + labels
        conf[iou < threshold] = 0

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc, conf, nms_threshold, conf_threshold):
        '''
        ToDo: decode()がCPU処理でボトルネックとなっているので高速化する。
              predict(20ms)に対してdecode()が20ms 〜 150msとなる場合がある。
              th以上の予測bboxが多い or overlapped bboxが多い場合に、その数に比例して遅くなる。
              単純にGPU処理化(cupy化)するだけだと、for文箇所がCPU処理なので遅い。
              従って、for文も含めGPU内部で処理する必要がある。
              実装の方針は、以下が参考になる：
              https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
              https://github.com/chainer/chainercv/blob/3f2a580cf15c50f115681c6ebfd0da30e71e5cfb/chainercv/utils/bbox/non_maximum_suppression.py
              ただし、chainer6.0.0 & cupy6.0.0以上が必要。
              現状chainer3.2.0 & cupy2.2なので、上記実装は使えない。
              一方でchainer&cupyを最新化しようとすると既存コードで動かない箇所が出てくるので
              システム刷新の際にchainer & cupyも最新化の上、ソース改修行うこと。
        '''
        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2        # 重心から左上の座標へ変換
        boxes[:, 2:] += boxes[:, :2]
        conf = np.exp(conf)
        conf /= conf.sum(axis=1, keepdims=True)
        scores = conf[:, 1:]

        all_boxes = list()
        all_labels = list()
        all_scores = list()

        for label in range(scores.shape[1]):
            mask = scores[:, label] >= conf_threshold
            label_boxes = boxes[mask]
            label_scores = scores[mask, label]

            selection = np.zeros(len(label_scores), dtype=bool)
            for i in label_scores.argsort()[::-1]:
                iou = matrix_iou(
                    label_boxes[np.newaxis, i],
                    label_boxes[selection])
                if (iou > nms_threshold).any():
                    continue
                selection[i] = True
                all_boxes.append(label_boxes[i])
                all_labels.append(label)
                all_scores.append(label_scores[i])

        return np.stack(all_boxes), np.stack(all_labels), np.stack(all_scores)
