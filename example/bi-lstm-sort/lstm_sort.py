# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import pickle
import logging

import numpy as np
import mxnet as mx

from sort_io import BucketSentenceIter, default_build_vocab
from lstm import build_lstm_network

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 100
    num_hidden = 300
    num_embed = 512
    num_lstm_layer = 2

    buckets=[]
    num_epoch = 10
    learning_rate = 0.1
    momentum = 0.9

    contexts = [mx.context.gpu(i) for i in range(1)]
    #contexts = [mx.context.cpu()]

    vocab = default_build_vocab("./data/sort.train.txt")

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter("./data/sort.train.txt", vocab, buckets, batch_size, init_states)
    data_val = BucketSentenceIter("./data/sort.valid.txt", vocab, buckets, batch_size, init_states)
    seq_len = data_train.seq_len

    # save parameters for testing
    with open('params.pickle', 'w') as f:
        pickle.dump([vocab, num_hidden, num_lstm_layer, seq_len, num_epoch], f)

    symbol = build_lstm_network(seq_len, len(vocab), num_hidden=num_hidden, num_embed=num_embed, num_label=len(vocab))

    model = mx.model.FeedForward(ctx=contexts, symbol=symbol, num_epoch=num_epoch,
                                 learning_rate=learning_rate, momentum=momentum, wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'data_train is', data_train.provide_label
    model.fit(X=data_train, eval_data=data_val, eval_metric = mx.metric.np(Perplexity),
		      batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

    model.save("sort")
