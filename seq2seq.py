#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''
import traceback
import random
import numpy as np
import theano
import theano.tensor as T
import lasagne
import scipy.io
from lasagne.layers import *
import cPickle
from recurrent import LSTMLayer as MyLSTMLayer
from recurrent import SliceLayer as MySliceLayer

n_hidden = 300
max_seq_len = 67
learning_rate = 5e-3
grad_clip = 10

VOCAB_SOURCE = 3437
VOCAB_TARGET = 3122

def build_encoder_network(num_inputs, num_hidden):
    input  = T.TensorType('float32', [None]*3)('input')
    B, L = input.shape[0:2]

    l_in = InputLayer((None, max_seq_len, num_inputs))

    l_mask = InputLayer(shape=(None, max_seq_len))

    l_enc = MyLSTMLayer(l_in, num_hidden, mask_input=l_mask, grad_clipping=grad_clip,
                      nonlinearity=lasagne.nonlinearities.rectify, only_return_final=True)

    params = lasagne.layers.get_all_params(l_enc)

    hid_out, _ = lasagne.layers.get_output(l_enc, {l_in: input})

    tvars = [input, l_mask.input_var]

    return hid_out, tvars, theano.function(tvars, hid_out), params

def build_decoder_network(tvars, enc_net_params, encoded_msg, num_hidden, num_outputs):

    num_inputs = num_outputs

    input  = T.TensorType('float32', [None]*3)('input')
    target = T.TensorType('int64',   [None]*2)('target')
    hid_init = T.TensorType('float32', [None]*2)('hid_init')
    # cell_init = T.TensorType('float32', list(encoded_msg.broadcastable))('cell_init')

    B, L = input.shape[0:2]

    target_reshaped = target.flatten()

    l_in = InputLayer((None, max_seq_len, num_inputs))

    l_mask = InputLayer(shape=(None, max_seq_len))

    l_dec = MyLSTMLayer(l_in, num_hidden, mask_input=l_mask,
                        grad_clipping=grad_clip,
                        hid_init=encoded_msg,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        only_return_final=False)

    l_slice = MySliceLayer(l_dec, indices=0, axis=0)

    l_reshape = ReshapeLayer(l_slice, (B*L, n_hidden))

    l_dense = DenseLayer(l_reshape, num_outputs, nonlinearity=lasagne.nonlinearities.softmax)

    l_out = l_dense

    # Get all the params
    params = enc_net_params + lasagne.layers.get_all_params(l_out)

    # Get output, reshape, and pass it to softmax
    hid_out, cell_out = lasagne.layers.get_output(l_dec, {l_in: input})
    hid_out = hid_out[:, -1, :]
    cell_out = cell_out[:, -1, :]
    output = lasagne.layers.get_output(l_out, {l_in: input})

    posterior = output.reshape((B, L, -1), ndim=3)

    nz_ix = target_reshaped.nonzero()
    output = output[nz_ix]
    target_reshaped = target_reshaped[nz_ix]

    # loss function
    loss = T.mean(T.nnet.categorical_crossentropy(output, target_reshaped))

    updates = lasagne.updates.adagrad(loss, params, learning_rate)

    tvars2 = [input, l_mask.input_var, target]

    # For train-set
    print "Compiling train ..."
    train = theano.function(tvars + tvars2, loss, updates=updates)

    # For dev-set
    print "Compiling predict_and_get_loss..."
    predict_and_get_loss = theano.function([hid_init] + tvars2, [posterior, hid_out, loss],
                                           givens={encoded_msg: hid_init},
                                           updates={l_dec.cell_init: cell_out}
                                           )
    
    # For test-set
    print "Compiling predict..."
    predict = theano.function([hid_init, input, l_mask.input_var], [posterior, hid_out],
                              givens={encoded_msg: hid_init},
                              updates={l_dec.cell_init: cell_out}
                              )

    return train, predict_and_get_loss, predict

def load_data(split):
    asca = np.ascontiguousarray

    src_mat = scipy.io.loadmat('data/%s.src.mat' % (split))
    src, src_mask = asca(src_mat['input']), asca(src_mat['mask'].astype(np.float32))

    try:
        tgt_mat = scipy.io.loadmat('data/%s.tgt.mat' % (split))
        tgt, tgt_mask = asca(tgt_mat['input']), asca(tgt_mat['mask'].astype(np.float32))
    except:
        tgt, tgt_mask = None, None

    return src, src_mask, tgt, tgt_mask

def get_batches(data, batch_size, randshuf=True, num_epochs=1):
    src, src_mask, tgt, tgt_mask = data

    N = int(np.ceil(float(src.shape[0]) / float(batch_size)))

    batch_indices = range(N)

    for epoch in range(num_epochs):
        if randshuf:
            random.shuffle(batch_indices)

        for i in batch_indices:
            begin = i*batch_size
            end = min(src.shape[0], (i+1)*batch_size)

            x = src[begin:end, :]
            y = tgt[begin:end, :] if tgt is not None else None

            x_mask = src_mask[begin:end, :]
            y_mask = tgt_mask[begin:end, :] if tgt is not None else None

            x_one_hot = to_one_hot(x, VOCAB_SOURCE)

            if tgt is not None:
                y_one_hot = to_one_hot(y, VOCAB_TARGET)
                y_one_hot = y_one_hot[:, range(-1, -1 + y_one_hot.shape[1]), :]
                y_one_hot[:, 0, :] = 0
            else:
                y_one_hot = None

            yield x, x_mask, x_one_hot, y, y_mask, y_one_hot

# x.shape = (batch_size, length)
def to_one_hot(x, vocab_size):
    one_hot = np.zeros(x.shape + (vocab_size,), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot[i, j, x[i, j]] = 1
    return one_hot

def gen_random_data():
    x = (np.random.rand(batch_size, max_seq_len) * VOCAB_SOURCE).astype(np.int64)
    x_mask = (np.random.rand(batch_size, max_seq_len) > 0.5).astype(np.float32)
    y = (np.random.rand(batch_size, max_seq_len) * VOCAB_TARGET).astype(np.int64)
    y_mask = x_mask.copy()

    x_one_hot = to_one_hot(x, VOCAB_SOURCE)
    y_one_hot = to_one_hot(y, VOCAB_TARGET)

    # print_shape_and_dtype(x, x_mask, x_one_hot, y, y_mask, y_one_hot)
    return x, x_mask, x_one_hot, y, y_mask, y_one_hot

def print_shape_and_dtype(x, x_mask, x_one_hot, y, y_mask, y_one_hot): 
    print "x.shape = {}, x.dtype = {}".format(x.shape, x.dtype)
    print "x_one_hot.shape = {}".format(x_one_hot.shape)
    print "x_mask.shape = {}, x_mask.dtype = {}".format(x_mask.shape, x_mask.dtype)

    print "y.shape = {}, y.dtype = {}".format(y.shape, y.dtype)
    print "y_one_hot.shape = {}".format(y_one_hot.shape)
    print "y_mask.shape = {}, y_mask.dtype = {}".format(y_mask.shape, y_mask.dtype)

def build_encoder_decoder_network():
    encoded_tvar, tvars, encode, enc_net_params = build_encoder_network(VOCAB_SOURCE, n_hidden)
    train, predict_and_get_loss, predict = build_decoder_network(tvars, enc_net_params, encoded_tvar, n_hidden, VOCAB_TARGET)
    return encode, train, predict_and_get_loss, predict

def loadVocab(path):
    return cPickle.load(open(path,'rb'))

if __name__ == '__main__':

    encode, train, predict_and_get_loss, predict = build_encoder_decoder_network()

    train_set = load_data("train")
    dev_set   = load_data("dev")
    test_set  = load_data("test")

    vocab = loadVocab("data/train.tgt.vocab.pickle")
    endIdx=vocab["</s>"]
    startIdx=vocab["<s>"]

    vocab_rev = dict((v,k) for k,v in vocab.iteritems())
    #print vocab_rev

    print "Training ..."
    batches = get_batches(train_set, batch_size=192, num_epochs=10, randshuf=False)
    for i, (x, x_mask, x_one_hot, y, y_mask, y_one_hot) in enumerate(batches):
        #for j in range(100):
        loss = train(x_one_hot, x_mask, y_one_hot, y_mask, y)
        print "batch #{:4d}, loss = {}".format(i, loss)
        #break

    '''
    print "Testing on dev-set..."
    batches = get_batches(dev_set, batch_size=1, randshuf=False)
    for i, (x, x_mask, x_one_hot, y, y_mask, y_one_hot) in enumerate(batches):

        encoded_msg = encode(x_one_hot, x_mask)

        T = x.shape[1]
        hid_prev = encoded_msg
        next_input = np.zeros((1, 1, VOCAB_TARGET), dtype=np.float32)
        mask = np.ones((1, 1), dtype=np.float32)
        loss = np.zeros((T))
        for t in range(T):
            posterior, hid_prev, loss[t] = predict_and_get_loss(hid_prev, next_input, mask, y[:, t:t+1])
            word_id = np.argmax(posterior)
            next_input = to_one_hot(np.array([[word_id]]), VOCAB_TARGET)

        print "#{:3d}, mean loss = {}".format(i, np.mean(loss))
    '''

    print "Testing on test-set..."
    batches = get_batches(train_set, batch_size=1, num_epochs=1, randshuf=False)
    f = open("translation.txt","w")
    for i, (x, x_mask, x_one_hot, y, y_mask, y_one_hot) in enumerate(batches):
        encoded_msg = encode(x_one_hot, x_mask)
        T = np.sum(x_mask[0, ])
        hid_prev = encoded_msg
        print hid_prev.shape
        next_input = np.zeros((1, 1, VOCAB_TARGET), dtype=np.float32)
        mask = np.ones((1, 1), dtype=np.float32)
        candidates = []

        isEndOfSent = False

        for t in range(int(T*1.5)):
            posterior, hid_prev = predict(hid_prev, next_input, mask)
            # print posterior
            word_id = np.argmax(posterior)
            print vocab_rev[word_id]
            #next_input = to_one_hot(np.array([[word_id]]), VOCAB_TARGET)
            #candidates.append(vocab_rev[word_id])
           
            #import ipdb; ipdb.set_trace()
            
            word_ids = np.argsort(posterior).flatten()[-10:][::-1]
            for w in word_ids:
                if w == startIdx and t!=0:
                    continue
                if t < T*0.5:
                    if word_id == endIdx:
                        continue
                if w == endIdx: # TODO
                    isEndOfSent = True
                next_input = to_one_hot(np.array([[w]]), VOCAB_TARGET)
                candidates.append(vocab_rev[w])
                break

            if isEndOfSent == True:
                break
            

        if not isEndOfSent:
            candidates.append("</s>")
        print " ".join(candidates)
        f.write(" ".join(candidates)+"\n")

        #break

    f.close()
