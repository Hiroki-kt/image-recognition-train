import numpy as np
import re
import sys

import chainer.links.caffe.caffe_function as cf

from ssdnet.ssd import _Normalize


def _rename(name):
    m = re.match(r'^conv(\d+)_([123])$', name)
    if m:
        i, j = list(map(int, m.groups()))
        if i >= 6:
            i += 2
        return 'conv{:d}_{:d}'.format(i, j)

    m = re.match(r'^fc([67])$', name)
    if m:
        return 'conv{:d}'.format(int(m.group(1)))

    if name == r'conv4_3_norm':
        return 'norm4'

    m = re.match(r'^conv4_3_norm_mbox_(loc|conf)$', name)
    if m:
        return '{:s}/0'.format(m.group(1))

    m = re.match(r'^fc7_mbox_(loc|conf)$', name)
    if m:
        return ('{:s}/1'.format(m.group(1)))

    m = re.match(r'^conv(\d+)_2_mbox_(loc|conf)$', name)
    if m:
        i, type_ = int(m.group(1)), m.group(2)
        if i >= 6:
            return '{:s}/{:d}'.format(type_, i - 4)

    return name


class _CaffeFunction(cf.CaffeFunction):

    def __init__(self, model_path, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print('loading weights from {:s} ... '.format(model_path))
        super(_CaffeFunction, self).__init__(model_path)

    def add_link(self, name, link):
        new_name = _rename(name)
        if self.verbose:
            print('{:s} -> {:s}'.format(name, new_name))
        super(_CaffeFunction, self).add_link(new_name, link)

    @cf._layer('Normalize', None)
    def _setup_normarize(self, layer):
        blobs = layer.blobs
        func = _Normalize(cf._get_num(blobs[0]))
        func.scale.data[:] = np.array(blobs[0].data)
        self.add_link(layer.name, func)

    @cf._layer('AnnotatedData', None)
    @cf._layer('Flatten', None)
    @cf._layer('MultiBoxLoss', None)
    @cf._layer('Permute', None)
    @cf._layer('PriorBox', None)
    def _skip_layer(self, _):
        pass


def load_caffe(model_path, verbose=False):
    return _CaffeFunction(model_path, verbose)
