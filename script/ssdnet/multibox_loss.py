# -*- coding: utf-8 -*-
import numpy as np

import chainer
import chainer.functions as F
from chainer import cuda
try:
    import cupy
except ImportError:
    cupy = None


def _elementwise_softmax_cross_entropy(x, t):
    assert x.shape[:-1] == t.shape
    p = F.reshape(
        F.select_item(F.reshape(x, (-1, x.shape[-1])), F.flatten(t)),
        t.shape)
    return F.logsumexp(x, axis=-1) - p


def _mine_hard_negative(loss, pos, k):
    xp = chainer.cuda.get_array_module(loss)
    loss = chainer.cuda.to_cpu(loss)
    pos = chainer.cuda.to_cpu(pos)
    rank = (loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
    hard_neg = rank < (pos.sum(axis=1) * k)[:, np.newaxis]
    return xp.array(hard_neg)


def multibox_loss(x_loc, x_conf, t_loc, t_conf, k):
    with chainer.cuda.get_device(t_conf.data):
        if cupy and isinstance(t_conf.data, cupy.cuda.memory.MemoryPointer):
            # for chainer v3+
            xp = chainer.cuda.get_array_module(t_conf)
            pos = t_conf > 0
        else:
            # old chianer version2
            xp = chainer.cuda.get_array_module(t_conf.data)
            pos = t_conf.data > 0

        # old version
        '''
        if xp.logical_not(pos).all():
            # 値が全て0の場合
            # return 0, 0
            return chainer.Variable(cuda.ndarray(0, dtype=np.float32)),\
                   chainer.Variable(cuda.ndarray(0, dtype=np.float32))
        '''
        # new version from chainerCV
        n_pos = pos.sum()       # number of positive enums
        if ( n_pos == 0 ):
            # 値が全て0の場合
            zero = chainer.Variable(xp.zeros((), dtype=np.float32))
            return zero, zero

        x_loc = F.reshape(x_loc, (-1, 4))
        t_loc = F.reshape(t_loc, (-1, 4))
        loss_loc = F.huber_loss(x_loc, t_loc, 1)
        loss_loc *= pos.flatten().astype(loss_loc.dtype)
        loss_loc = F.sum(loss_loc) / n_pos
            
        loss_conf = _elementwise_softmax_cross_entropy(x_conf, t_conf)
        hard_neg = _mine_hard_negative(loss_conf.data, pos, k)
        loss_conf *= xp.logical_or(pos, hard_neg).astype(loss_conf.dtype)
        loss_conf = F.sum(loss_conf) / n_pos

        return loss_loc, loss_conf
