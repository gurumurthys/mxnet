# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

class BiLSTMInferenceModel(object):
    def __init__(self, s, args_params, aux_params, num_hidden, seq_len, num_lstm_layer):
        self.sym = s
        batch_size = 1
        init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        
        data_shape = [("data", (batch_size, seq_len, ))]
        label_shape = [("softmax_label", (batch_size, seq_len, ))]

        input_shapes = dict(init_c + init_h + data_shape+label_shape)
        self.executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)

        for key in self.executor.arg_dict.keys():
            if key in args_params:
                args_params[key].copyto(self.executor.arg_dict[key])

        state_name = []
        for i in range(num_lstm_layer):
            state_name.append("l%d_init_c" % i)
            state_name.append("l%d_init_h" % i)

        self.states_dict = dict(zip(state_name, self.executor.outputs[1:]))
        self.input_arr = mx.nd.zeros(data_shape[0][1])

    def forward(self, input_data, new_seq=False):
        if new_seq == True:
            for key in self.states_dict.keys():
                self.executor.arg_dict[key][:] = 0.
        input_data.copyto(self.executor.arg_dict["data"])
        self.executor.forward()
        for key in self.states_dict.keys():
            self.states_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob
