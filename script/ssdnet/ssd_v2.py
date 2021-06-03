# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class _SSDVGG16_v2(chainer.Chain):
    '''
    pre-trained VGG16 model (fc reduced)
    '''
    mean = (104, 117, 123)
    variance = (0.1, 0.2)

    conv_init = {
        'initialW': initializers.GlorotUniform(),
        'initial_bias': initializers.Zero(),
    }
    norm_init = {
        'initial': initializers.Constant(20),
    }

    def __init__(self, labels):
        self.labels = labels
        self.n_classes = len(labels)

        super(_SSDVGG16_v2, self).__init__(
            # input, output and kernel dimensions.
            conv1_1=L.Convolution2D(None, 64, 3, pad=1, **self.conv_init),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1, **self.conv_init),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1, **self.conv_init),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1, **self.conv_init),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1, **self.conv_init),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1, **self.conv_init),
            norm4=L.BatchNormalization(512, decay=0.99),

            conv5_1=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),
            conv5_2=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),
            conv5_3=L.DilatedConvolution2D(
                None, 512, 3, pad=1, **self.conv_init),

            conv6=L.DilatedConvolution2D(
                None, 1024, 3, pad=6, dilate=6, **self.conv_init),
            conv7=L.Convolution2D(None, 1024, 1, **self.conv_init),
            # mitarai
            norm7=L.BatchNormalization(1024, decay=0.99),

            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )
        for ar in self.aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(
                None, n * 4, 3, pad=1, **self.conv_init))
            self.conf.add_link(L.Convolution2D(
                None, n * (self.n_classes + 1), 3, pad=1, **self.conv_init))

    def _features(self, x):
        ys = list()

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = self.conv4_3(h)
        h = self.norm4(h)
        h = F.relu(h)
        ys.append(h)
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = self.conv7(h)
        h = self.norm7(h)
        h = F.relu(h)
        ys.append(h)

        return ys

    def _multibox(self, xs):
        ys_loc = list()
        ys_conf = list()
        for i, x in enumerate(xs):
            loc = self.loc[i](x)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            ys_loc.append(loc)

            conf = self.conf[i](x)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(
                conf, (conf.shape[0], -1, self.n_classes + 1))
            ys_conf.append(conf)

        y_loc = F.concat(ys_loc, axis=1)
        y_conf = F.concat(ys_conf, axis=1)
        return y_loc, y_conf

    def __call__(self, x):
        return self._multibox(self._features(x))


class SSD300_v2(_SSDVGG16_v2):
    '''
    SSD Layers, started layer 8_1.
    SSD base conv-layer is _SSDVGG16_v2(#convs: 7)
    '''
    insize = 300
    grids = (38, 19, 10, 5, 3, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)]

    def __init__(self, labels):
        super(SSD300_v2, self).__init__(labels)

        self.add_link(
            'conv8_1', L.Convolution2D(None, 256, 1, **self.conv_init))
        self.add_link(
            'conv8_2',
            L.Convolution2D(None, 512, 3, stride=2, pad=1, **self.conv_init))
        self.add_link(
            'norm8',
            L.BatchNormalization(512, decay=0.99))

        self.add_link(
            'conv9_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv9_2',
            L.Convolution2D(None, 256, 3, stride=2, pad=1, **self.conv_init))
        self.add_link(
            'bat_norm9',
            L.BatchNormalization(256, decay=0.99))

        self.add_link(
            'conv10_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv10_2', L.Convolution2D(None, 256, 3, **self.conv_init))
        self.add_link(
            'norm10',
            L.BatchNormalization(256, decay=0.99))

        self.add_link(
            'conv11_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv11_2', L.Convolution2D(None, 256, 3, **self.conv_init))
        self.add_link(
            'norm11',
            L.BatchNormalization(256, decay=0.99))

    def _features(self, x):
        ys = super(SSD300_v2, self)._features(x)
        for i in range(8, 11 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = self['conv{:d}_2'.format(i)](h)
            h = self['norm{:d}'.format(i)](h)
            h = F.relu(h)
            ys.append(h)
        return ys


class SSD512_v2(_SSDVGG16_v2):

    insize = 512
    grids = (64, 32, 16, 8, 4, 2, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2, ))
    steps = [s / 512 for s in (8, 16, 32, 64, 128, 256, 512)]
    sizes = [s / 512 for s in
             (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)]

    def __init__(self, labels):
        super(SSD512_v2, self).__init__(labels)

        self.add_link(
            'conv8_1', L.Convolution2D(None, 256, 1, **self.conv_init))
        self.add_link(
            'conv8_2',
            L.Convolution2D(None, 512, 3, stride=2, pad=1, **self.conv_init))
        self.add_link(
            'norm8',
            L.BatchNormalization(512, decay=0.99))

        for i in range(9, 11 + 1):
            self.add_link(
                'conv{:d}_1'.format(i),
                L.Convolution2D(None, 128, 1, **self.conv_init))
            self.add_link(
                'conv{:d}_2'.format(i),
                L.Convolution2D(
                    None, 256, 3, stride=2, pad=1, **self.conv_init))
            self.add_link(
                'norm{:d}'.format(i),
                L.BatchNormalization(256, decay=0.99))

        self.add_link(
            'conv12_1', L.Convolution2D(None, 128, 1, **self.conv_init))
        self.add_link(
            'conv12_2',
            L.Convolution2D(None, 256, 4,  pad=1, **self.conv_init))
        self.add_link(
            'norm12',
            L.BatchNormalization(256, decay=0.99))

    def _features(self, x):
        ys = super(SSD512_v2, self)._features(x)
        for i in range(8, 12 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = self['conv{:d}_2'.format(i)](h)
            h = self['norm{:d}'.format(i)](h)
            h = F.relu(h)
            ys.append(h)
        return ys
