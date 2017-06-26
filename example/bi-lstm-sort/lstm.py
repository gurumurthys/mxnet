# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
from collections import namedtuple

import mxnet as mx
import numpy as np
import time
import math

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states", "forward_state", "backward_state",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])
def LSTM(name, layeridx, input_data, seq_len, num_hidden, reverse = False):
    param = LSTMParam(i2h_weight=mx.sym.Variable(name+"_i2h_weight"),
                              i2h_bias=mx.sym.Variable(name+"_i2h_bias"),
                              h2h_weight=mx.sym.Variable(name+"_h2h_weight"),
                              h2h_bias=mx.sym.Variable(name+"_h2h_bias"))
    hidden_vals = []
    state_vals = []
    lstm_state = LSTMState(c = mx.sym.Variable(name+"_init_c"), h = mx.sym.Variable(name+"_init_h"))
    for seqidx in range(seq_len):
        k = seqidx
        if(reverse):
            k = seq_len - seqidx - 1
        hidden = input_data[k]
        next_state = lstmcell(num_hidden, indata=hidden,
                          prev_state=lstm_state,
                          param=param,
                          seqidx=k, layeridx=layeridx, dropout=0)
        hidden = next_state.h
        lstm_state = next_state
	if(reverse):
		state_vals.insert(0, next_state)
		hidden_vals.insert(0, hidden)
	else:
		state_vals.append(next_state)
		hidden_vals.append(hidden)

    return hidden_vals, state_vals

def lstmcell(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    return LSTMState(c=next_c, h=next_h)

def build_lstm_network(seq_len, input_size, num_hidden, num_embed, num_label):
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    hiddenf, statesf = LSTM('l0', 0, wordvec, seq_len, num_hidden)
    hiddenb, statesb = LSTM('l1', 1, wordvec, seq_len, num_hidden, reverse=True)
        
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[hiddenf[i], hiddenb[i]], dim=1))

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))
    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm
