# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")

import pickle
import numpy as np
import mxnet as mx

from sort_io import default_build_vocab
from inference_model import BiLSTMInferenceModel

if __name__ == '__main__':
    contexts = [mx.context.gpu(i) for i in range(1)]
    with open('params.pickle') as f:  # Python 3: open(..., 'rb')
            vocab, num_hidden, num_lstm_layer, seq_len, num_epoch = pickle.load(f)

    rvocab = {}
    for k, v in vocab.items():
        rvocab[v] = k

    tks = sys.argv[1:]
    data = np.zeros((1, len(tks)))
    for k in range(len(tks)):
        data[0][k] = vocab[tks[k]]
    data = mx.nd.array(data)

    s, arg_params, aux_params = mx.model.load_checkpoint("sort", num_epoch)
    model = BiLSTMInferenceModel(s, arg_params, aux_params, num_hidden, seq_len, num_lstm_layer)
    prob =  model.forward(data)
    for k in range(len(tks)):        
        print(rvocab[np.argmax(prob, axis = 1)[k]])
