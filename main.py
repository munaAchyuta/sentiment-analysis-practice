import sys
import time
import math
import os
import numpy as np
import tensorflow as tf
from utils import batch_index, load_inputs_data_at, load_data_init, load_input_data_at
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn as sk


batch_size_init = 25
n_hidden_init = 300
learning_rate_init = 0.001
n_class_init = 3
l2_reg_init = 0.001
n_iter_init = 20
keep_prob1_init = 0.5
keep_prob2_init = 0.5
keep_prob0_init = 0.8
method = 'BiLSTM'
hopnum = 0
num_layers = 2

if len(sys.argv)>1:
    batch_size_init=int(sys.argv[1])
    n_hidden_init = int(sys.argv[2])
    learning_rate_init = float(sys.argv[3])
    n_class_init = int(sys.argv[4])
    l2_reg_init = float(sys.argv[5])
    n_iter_init = int(sys.argv[6])
    keep_prob1_init = float(sys.argv[7])
    keep_prob2_init = float(sys.argv[8])
    method = sys.argv[9]
    hopnum = int(sys.argv[10])


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('debugmode', '1', 'is debug mode: ')

tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', batch_size_init, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', n_hidden_init, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', learning_rate_init, 'learning rate')
tf.app.flags.DEFINE_integer('num_layers', num_layers, 'number of layers')
tf.app.flags.DEFINE_integer('n_class', n_class_init, 'number of distinct class')
tf.app.flags.DEFINE_float('l2_reg', l2_reg_init, 'l2 regularization')
tf.app.flags.DEFINE_integer('n_iter', n_iter_init, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', keep_prob1_init, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', keep_prob2_init, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob0', keep_prob0_init, 'dropout keep prob')


tf.app.flags.DEFINE_string('dataset', 'data/ms_St1k_train_data.txt', 'training file')
tf.app.flags.DEFINE_string('devset', 'data/dev_data.txt', 'development file')
tf.app.flags.DEFINE_string('testset', 'data/ms_St500_test_data.txt', 'testing file')#ms_St500_test_data.txt
tf.app.flags.DEFINE_string('embedding_file_path', 'data/glove.6B.300d.txt', 'embedding file')
tf.app.flags.DEFINE_string('entity_id_file_path', 'data/ms_St_tag_data.txt', 'entity-id mapping file')
tf.app.flags.DEFINE_string('method', method.split('-')[0], 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('t', 'last', 'model type: ')
tf.app.flags.DEFINE_integer('hopnum', hopnum, 'model type: ')
tf.app.flags.DEFINE_string('predict', 'no', 'process type: ')


class LSTM(object):

    def __init__(self,vocab_size, embedding_dim=100, pos_embedding_dim=100, batch_size=64, n_hidden=100, learning_rate=0.01,
                 n_class=3, max_sentence_len=140, l2_reg=0.,  n_iter=100, type_='', num_layers=1):
        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.num_layers = num_layers
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.n_iter = n_iter
        self.type_ = type_
        self.Embed = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                                 trainable=True, name="embedd")
        self.pos_Embed = tf.Variable(tf.constant(0.0, shape=[pos_embedding_dim, pos_embedding_dim]),
                                 trainable=True, name="pos_embedd")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        self.pos_embedding_placeholder = tf.placeholder(tf.float32, [pos_embedding_dim, pos_embedding_dim])
        self.embedding_init = self.Embed.assign(self.embedding_placeholder)
        self.pos_embedding_init = self.pos_Embed.assign(self.pos_embedding_placeholder)
        self.sentence_len = tf.placeholder(tf.int32, [None])

        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)
        self.keep_prob0 = tf.placeholder(tf.float32)

        self.restore=False
        self.needdev=False
        if self.restore:
            self.needdev = False
        self.showtrainerr=True

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
            self.pos_x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='pos_x')
            self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')
            self.sen_len = tf.placeholder(tf.int32, None, name='sen_len')
            self.entity_id = tf.placeholder(tf.int32, None, name='entity_id')
            self.e_loc = tf.placeholder(tf.float32, [None, self.max_sentence_len, 1], name='e_loc')

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.get_variable(
                    name='softmax_w',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.get_variable(
                    name='softmax_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        self.W = tf.get_variable(
            name='W',
            shape=[self.n_hidden + self.embedding_dim, self.n_hidden + self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.w = tf.get_variable(
            name='w',
            shape=[self.n_hidden + self.embedding_dim, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.b = tf.get_variable(
            name='b',
            shape=[1, self.max_sentence_len],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.We = tf.get_variable(
            name='We',
            shape=[self.max_sentence_len, self.max_sentence_len],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wa = tf.get_variable(
            name='Wa',
            shape=[self.max_sentence_len, self.max_sentence_len],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wea = tf.get_variable(
            name='Wea',
            shape=[self.embedding_dim, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wetat = tf.get_variable(
            name='Wetat',
            shape=[self.n_hidden, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Watat = tf.get_variable(
            name='Watat',
            shape=[self.n_hidden, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wp = tf.get_variable(
            name='Wp',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wx = tf.get_variable(
            name='Wx',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wdmne = tf.get_variable(
            name='Wdmne',
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wdmna = tf.get_variable(
            name='Wdmna',
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wdmnr = tf.get_variable(
            name='Wdmnr',
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wdmn = tf.get_variable(
            name='Wdmn',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Bdmn = tf.get_variable(
            name='Bdmn',
            shape=[1, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wdmn_e = tf.get_variable(
            name='Wdmn_e',
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wdmn_a = tf.get_variable(
            name='Wdmn_a',
            shape=[self.embedding_dim, self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )

    def dynamic_rnn(self, cell, inputs, length, max_len, scope_name, out_type='all'):
        outputs, state = tf.nn.dynamic_rnn(
            cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        ) 
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)
        elif out_type == 'all_avg':
            outputs = LSTM.reduce_mean(outputs, length)
        if FLAGS.method == "SLSTM":
            return state.h
        return outputs
    
    def SLSTM(self, inputs, type_='last'):
        print('I am SLSTM.')
        batch_size = tf.shape(inputs)[0]
        in_t = tf.nn.dropout(inputs, keep_prob=self.keep_prob0)
        cell = tf.nn.rnn_cell.LSTMCell
        hiddens = self.dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT', "all")

        return LSTM.softmax_layer(hiddens, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)
    
    def MLSTM(self, inputs, num_layers=1):
        print('I am MLSTM or mulit layer LSTM. & num_layers : ',self.num_layers)
        batch_size = tf.shape(inputs)[0]
        in_t = tf.nn.dropout(inputs, keep_prob=self.keep_prob0)
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(self.n_hidden,initializer=tf.orthogonal_initializer(),reuse=reuse)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(self.num_layers)])
        drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob = self.keep_prob1)
        outputs, _ = tf.nn.dynamic_rnn(drop, in_t, dtype = tf.float32)
        #hiddens = self.dynamic_rnn(drop, in_t, self.sen_len, self.max_sentence_len, 'AT', "all")
        outputs = outputs[:, -1]
        return LSTM.softmax_layer(outputs, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)
    
    def BiRNN(self, inputs, num_layers=1):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        x = tf.unstack(inputs, self.n_hidden, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get BiRNN cell output
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        
        #outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        # Concat the forward and backward outputs
        #output = tf.concat(outputs,2)

        # Linear activation, using rnn inner loop last output
        #return tf.matmul(outputs[-1], weights) + biases
        return outputs[-1]
    
    def BiLSTM(self, inputs, type_='last'):
        print('I am BiLSTM.')
        batch_size = tf.shape(inputs)[0]
        in_t = tf.nn.dropout(inputs, keep_prob=self.keep_prob0)
        hiddens = BiRNN(in_t)

        return LSTM.softmax_layer(hiddens, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    def AT(self, inputs, entity, aspect, type_='last'):
        print('I am AT.')
        batch_size = tf.shape(inputs)[0]
        entity0 = tf.reshape(entity, [-1, 1, self.embedding_dim])
        entity = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * entity0
        aspect0 = tf.reshape(aspect, [-1, 1, self.embedding_dim])
        aspect = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * aspect0
        in_t = tf.concat([inputs, entity, aspect],2)

        in_t = tf.nn.dropout(in_t, keep_prob=self.keep_prob1)
        cell = tf.nn.rnn_cell.LSTMCell
        hiddens = self.dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT', "all")

        h_t = tf.reshape(tf.concat([hiddens, entity, aspect],2), [-1, self.n_hidden + 2*self.embedding_dim])

        M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
        alpha = LSTM.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), self.sen_len, self.max_sentence_len)

        self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])

        r = tf.reshape(tf.matmul(alpha, hiddens), [-1, self.n_hidden])
        index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
        hn = tf.gather(tf.reshape(hiddens, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(hn, self.Wx))

        return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    def DMN(self, inputs, entity, aspect, hopnum=5):
        print('I am DMN.')
        batch_size = tf.shape(inputs)[0]
        inputs = inputs*self.e_loc*self.a_loc

        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        for i in range(hopnum):
            entity0 = tf.reshape(entity, [-1, 1, self.embedding_dim])
            aspect0 = tf.reshape(aspect, [-1, 1, self.embedding_dim])
            entity1 = inputs * entity0
            aspect1 = inputs * aspect0
            h_t = tf.reshape(tf.concat([inputs, entity1, aspect1], 2), [-1, self.n_hidden + 2 * self.embedding_dim])

            M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
            alpha = LSTM.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), self.sen_len, self.max_sentence_len)

            self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])

            r = tf.reshape(tf.matmul(alpha, inputs), [-1, self.n_hidden])
            entity = entity + r
            aspect = aspect + r
        h = tf.matmul(tf.reshape(tf.concat([entity, aspect], 1), [-1, self.n_hidden * 2]), self.Wdmn)
        b = tf.ones([batch_size, self.n_hidden]) * self.Bdmn
        h = h + b
        return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    def CEA(self, inputs, entity, hopnum=3):
        print('I am CEA.')
        batch_size = tf.shape(inputs)[0]
        entity0 = tf.reshape(entity, [-1, 1, self.embedding_dim])
        entity1 = inputs * entity0
        in_t = tf.concat([inputs, entity1], 2)
        in_t = in_t * self.e_loc
        in_t = tf.nn.dropout(in_t, keep_prob=self.keep_prob0)
        cell = tf.nn.rnn_cell.LSTMCell
        hiddens = self.dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT', "all")

        for i in range(hopnum):
            entity0 = tf.reshape(entity, [-1, 1, self.embedding_dim])
            entity1 = hiddens * entity0
            h_t = tf.reshape(tf.concat([hiddens, entity1], 2), [-1, self.n_hidden + self.embedding_dim])

            M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
            selfb = tf.ones([batch_size,self.max_sentence_len]) * self.b
            M = tf.reshape(M, [-1, self.max_sentence_len])
            M = M + selfb
            M = tf.reshape(M, [-1, 1, self.max_sentence_len])
            alpha = LSTM.softmax(M, self.sen_len, self.max_sentence_len)
            self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])
            r = tf.reshape(tf.matmul(alpha, hiddens), [-1, self.n_hidden])
            entity = entity + r

        h = tf.matmul(tf.reshape(entity,[-1,self.n_hidden]),self.Wdmn)
        h = tf.nn.dropout(h, keep_prob=self.keep_prob2)
        b = tf.ones([batch_size,self.n_hidden])*self.Bdmn
        h = h + b
        return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    @staticmethod
    def softmax_layer(inputs, weights, biases, keep_prob):
        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
            predict = tf.matmul(outputs, weights) + biases
            predict = tf.nn.softmax(predict)
        return predict

    @staticmethod
    def reduce_mean(inputs, length):
        """
        :param inputs: 3-D tensor
        :param length: the length of dim [1]
        :return: 2-D tensor
        """
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs

    @staticmethod
    def softmax(inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def run(self, PreEmbedding, word2id, p2id, p2v):
        atime=time.time()
        inputs = tf.nn.embedding_lookup(PreEmbedding, self.x)
        entity = tf.nn.embedding_lookup(PreEmbedding, self.entity_id)
        if FLAGS.method == "CEA":
            prob = self.CEA(inputs, entity, hopnum = FLAGS.hopnum)
        if FLAGS.method == "SLSTM":
            prob = self.SLSTM(inputs)
        if FLAGS.method == "MLSTM":
            prob = self.MLSTM(inputs)
        if FLAGS.method == "BiLSTM":
            prob = self.BiLSTM(inputs)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if FLAGS.method == 'CEA':
                reg_loss = tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.W) \
                           + tf.nn.l2_loss(self.Wdmn) + tf.nn.l2_loss(self.Bdmn) + tf.nn.l2_loss(self.b) \
                           + tf.nn.l2_loss(self.weights['softmax']) + tf.nn.l2_loss(self.biases['softmax'])
                if FLAGS.hopnum == 0:
                    reg_loss = tf.nn.l2_loss(self.Wdmn) + tf.nn.l2_loss(self.Bdmn) + tf.nn.l2_loss(self.weights['softmax']) + tf.nn.l2_loss(self.biases['softmax'])
                reg_loss = reg_loss * self.l2_reg
                cost = - tf.reduce_mean(tf.cast(self.y, tf.float32) * tf.log(prob)) + reg_loss
            elif FLAGS.method in ["SLSTM","MLSTM","BiLSTM"]:
                reg_loss = tf.nn.l2_loss(self.weights['softmax']) + tf.nn.l2_loss(self.biases['softmax'])
                reg_loss = reg_loss * self.l2_reg
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.log(prob), labels=tf.cast(self.y, tf.float32))) + reg_loss
            else:
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.log(prob), labels=tf.cast(self.y, tf.float32))) 

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            true_y = tf.argmax(self.y, 1)
            pred_y = tf.argmax(prob, 1)
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess.run(init)
            
            # `sess.graph` provides access to the graph used in a `tf.Session`.
            #writer = tf.summary.FileWriter("tmp/log/", sess.graph)

            dt_x, dt_sen_len, dt_entity, dt_y, dt_yvalue, idlist, dt_pos, dt_eloc = load_inputs_data_at(
                FLAGS.dataset,
                word2id,
                self.max_sentence_len,
                p2id,
            )

            if self.needdev:
                """ shuffle the train set and split the train set into train and dev sets"""
                sss = StratifiedShuffleSplit(dt_yvalue, 1, test_size=0.1, random_state=0)
                print('len of sss',len(sss))
                for train_index, test_index in sss:
                    print("TRAIN:", len(train_index), "TEST:", len(test_index))
                    tr_x = dt_x[train_index]
                    te_x = dt_x[test_index]
                    tr_y = dt_y[train_index]
                    te_y = dt_y[test_index]
                    tr_sen_len = dt_sen_len[train_index]
                    te_sen_len = dt_sen_len[test_index]
                    tr_entity = dt_entity[train_index]
                    te_entity = dt_entity[test_index]
                    tr_aspect = dt_aspect[train_index]
                    te_aspect = dt_aspect[test_index]
                    tr_id = [idlist[x] for x in train_index]
                    te_id = [idlist[x] for x in test_index]
                    tr_pos = dt_pos[train_index]
                    te_pos = dt_pos[test_index]
                    tr_eloc = dt_eloc[train_index]
                    te_eloc = dt_eloc[test_index]
                    tr_aloc = dt_aloc[train_index]
                    te_aloc = dt_aloc[test_index]

                    ftrain = open('data/babycare/dataset0824-train.txt','w',encoding='utf8')
                    ftrain.write('\n'.join(tr_id))
                    ftrain.close()

                    ftrain = open('data/babycare/dataset0824-test.txt', 'w', encoding='utf8')
                    ftrain.write('\n'.join(te_id))
                    ftrain.close()

            else:
                tr_x, tr_y, tr_sen_len, tr_entity, tr_pos, tr_eloc = \
                    dt_x, dt_y, dt_sen_len, dt_entity, dt_pos, dt_eloc
                te_x, te_sen_len, te_entity, te_y, te_yvalue, idlist, te_pos, te_eloc = load_inputs_data_at(
                    FLAGS.testset,
                    word2id,
                    self.max_sentence_len,
                    p2id,
                    case='test'
                )

            del dt_x
            del dt_y
            del dt_sen_len
            del dt_entity
            del idlist
            del dt_pos
            del dt_eloc

            prtstr = "Configs:{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(self.batch_size, self.n_hidden,
                                                                           self.learning_rate,
                                                                           self.n_class, self.l2_reg, self.n_iter,
                                                                           keep_prob1_init, keep_prob2_init, method)
            prtstr2 = ''

            sess.run([self.embedding_init,self.pos_embedding_init], feed_dict={self.embedding_placeholder: PreEmbedding,self.pos_embedding_placeholder:p2v})

            savefilename = 'ME-ABSA-AAAI-' + FLAGS.method
            if self.restore:
                print("inside restore : ")
                saver.restore(sess, './models/'+savefilename)
                acc, loss, cnt = 0., 0., 0
                ctime = time.time()
                if FLAGS.predict == 'yes':
                    while True:
                        proc_text = input("Enter processed text : ")
                        proc_entity = input("Enter entity : ")
                        predset = {"text":proc_text,"entity":proc_entity}
                        de_x, de_sen_len, de_entity, de_y, de_pos, de_eloc = load_input_data_at(
                             predset,word2id,self.max_sentence_len,p2id,case='test')

                        for test, num in self.get_batch_data(de_x,de_pos, de_sen_len, de_y, de_entity, 1000, 1.0, 1.0, 1.0,de_eloc, False):
                            _prob, _loss, _acc= sess.run([prob, cost, accuracy],feed_dict=test)
                            acc += _acc
                            cnt += num
                        print("prediction ::: ")
                        print("prob : ",np.around(_prob, decimals = 3))
                        print("max prob : ",np.max(np.around(_prob, decimals = 3)[0]))
                        print("acc : ",acc)
                        print("cnt : ",cnt)
                        proc_flag = input("do you want to continue : ")
                        if proc_flag in ['n','N','NO','No','no']:
                            import sys;sys.exit()
                if FLAGS.method == "SLSTM":
                    for test, num in self.get_batch_data(te_x,te_pos, te_sen_len, te_y, te_entity, 1000, 1.0, 1.0, 1.0,te_eloc, False):
                        _loss, _acc, _step, ty, py = sess.run([cost, accuracy, global_step, true_y, pred_y],
                                                                    feed_dict=test)
                        acc += _acc
                        loss += _loss * num
                        cnt += num
                else:
                     for test, num in self.get_batch_data(te_x,te_pos, te_sen_len, te_y, te_entity, 1000, 1.0, 1.0, 1.0,te_eloc, False):
                        if FLAGS.hopnum==0:
                            _loss, _acc, _step, ty, py = sess.run([cost, accuracy, global_step, true_y, pred_y],
                                                                    feed_dict=test)
                        else:
                            _loss, _acc, _step, alpha, ty, py = sess.run([cost, accuracy, global_step, self.alpha, true_y, pred_y],
                                                                    feed_dict=test)
                        acc += _acc
                        loss += _loss * num
                        cnt += num

                print('all samples={}, correct prediction={}'.format(cnt, acc))
                print('mini-batch loss={:.6f}, test acc={:.6f}'.format(loss / cnt, acc / cnt))
                print("Precision", sk.metrics.precision_score(ty, py, average='micro'))
                print("Recall", sk.metrics.recall_score(ty, py, average='micro'))
                print("f1_score", sk.metrics.f1_score(ty, py, average='micro'))
                print("classification_report ",sk.metrics.classification_report(y_true=ty, y_pred=py, target_names=['0','1','2']))
            else:
                max_acc = 0.

                batch_count = 0
                tr_acc_li = []
                tr_loss_li = []
                te_acc_li = []
                te_loss_li = []
                for i in range(self.n_iter):
                    for train, _ in self.get_batch_data(tr_x, tr_pos, tr_sen_len, tr_y, tr_entity, \
                                                        self.batch_size, FLAGS.keep_prob0, FLAGS.keep_prob1, FLAGS.keep_prob2,tr_eloc,True):
                        _, step = sess.run([optimizer, global_step], feed_dict=train)
                        batch_count += 1
                        if batch_count%100==0:
                            print(batch_count * 100.0 * self.batch_size / len(tr_y) / self.n_iter,'% at iter',i)


                    acc, loss, cnt = 0., 0., 0
                    flag = True

                    ctime = time.time()
                    sb=''
                    if FLAGS.method == "SLSTM":
                        for test, num in self.get_batch_data(te_x,te_pos, te_sen_len, te_y, te_entity, 1000, 1.0, 1.0, 1.0,te_eloc, False):
                            _loss, _acc, _step, ty, py = sess.run([cost, accuracy, global_step, true_y, pred_y],
                                                                    feed_dict=test)

                            acc += _acc
                            loss += _loss * num
                            cnt += num
                            if flag:
                                flag = False

                            sb+='\n'.join([str(pyyy) for pyyy in py]) + '\n'
                    else:
                        for test, num in self.get_batch_data(te_x,te_pos, te_sen_len, te_y, te_entity, 1000, 1.0, 1.0, 1.0,te_eloc, False):
                            if FLAGS.hopnum==0:
                                _loss, _acc, _step, ty, py = sess.run([cost, accuracy, global_step, true_y, pred_y],
                                                                    feed_dict=test)
                            else:
                                _loss, _acc, _step, alpha, ty, py = sess.run([cost, accuracy, global_step, self.alpha, true_y, pred_y],
                                                                    feed_dict=test)

                            acc += _acc
                            loss += _loss * num
                            cnt += num
                            if flag:
                                flag = False

                            sb+='\n'.join([str(pyyy) for pyyy in py]) + '\n'
                    
                    print('all samples={}, correct prediction={}'.format(cnt, acc))
                    print('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, loss / cnt, acc / cnt))
                    prtstr += "{}\t{}\t{}\n".format(i, loss / cnt, acc / cnt)
                    te_loss_li.append(loss / cnt)
                    te_acc_li.append(acc / cnt)

                    if acc / cnt > max_acc:
                        max_acc = acc / cnt
                    testacc = acc/cnt
                    
                    if self.showtrainerr:
                        acc, loss, cnt = 0., 0., 0
                        for test, num in self.get_batch_data(tr_x, tr_pos, tr_sen_len, tr_y, tr_entity, 1000, 1, 1, 1,tr_eloc,is_shuffle=False):
                            _loss, _acc, _step, ty, py = sess.run([cost, accuracy, global_step, true_y, pred_y],feed_dict=test)
                            acc += _acc
                            loss += _loss * num
                            cnt += num
                            if flag:
                                flag = False
                        print('all samples={}, correct prediction in train={}'.format(cnt, acc))
                        print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}'.format(i, loss / cnt, acc / cnt))
                        print("Precision", sk.metrics.precision_score(ty, py, average='micro'))
                        print("Recall", sk.metrics.recall_score(ty, py, average='micro'))
                        print("f1_score", sk.metrics.f1_score(ty, py, average='micro'))
                        #print("classification_report ",sk.metrics.classification_report(y_true=ty, y_pred=py, target_names=['0','1','2']))

                        prtstr += "{}\t{}\t{}\n".format(i, loss / cnt, acc / cnt)
                        tr_loss_li.append(loss / cnt)
                        tr_acc_li.append(acc / cnt)
                    
                    
                    if i==self.n_iter-1 or testacc < 0.4:
                        btime = time.time()
                        prtstr2 += 'mathod=\t{}\tacc=\t{}\tLearning_rate=\t{}\titer_num=\t{}\tbatch_size=\t{}\thidden_num=\t{}\tl2=\t{}\ttraintime=\t{}\ttesttime=\t{}\thopnum=\t{}\tmaxacc=\t{}\n'.format(
                                    FLAGS.method,
                                    str(testacc),
                                    self.learning_rate,
                                    self.n_iter,
                                    self.batch_size,
                                    self.n_hidden,
                                    self.l2_reg,
                                    str(btime-atime),
                                    str(btime-ctime),
                                    str(FLAGS.hopnum),
                                    str(max_acc))
                        break
                print('Optimization Finished! Max acc={}\n{}'.format(max_acc,prtstr2))
                
                acc = tr_acc_li
                val_acc = te_acc_li

                loss = tr_loss_li
                val_loss = te_loss_li
                
                # Create count of the number of epochs
                epoch_count = range(1, len(acc) + 1)

                plt.figure(figsize=(8, 8))
                plt.subplot(2, 1, 1)
                plt.plot(epoch_count, acc, label='Training Accuracy')
                plt.plot(epoch_count, val_acc, label='Validation Accuracy')
                plt.legend(loc='lower right')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                #plt.ylim([min(plt.ylim()),1])
                plt.title('Training and Validation Accuracy')

                plt.subplot(2, 1, 2)
                plt.plot(epoch_count, loss, label='Training Loss')
                plt.plot(epoch_count, val_loss, label='Validation Loss')
                plt.legend(loc='upper right')
                plt.xlabel('Epoch')
                plt.ylabel('Cross Entropy')
                #plt.ylim([0,max(plt.ylim())])
                plt.title('Training and Validation Loss')
                #plt.show()
                plt.savefig('models/plot.png')   # save the figure to file
                plt.close()
                
                '''
                # Create count of the number of epochs
                epoch_count = range(1, len(acc) + 1)

                # Visualize loss history
                plt.plot(epoch_count, loss, 'r--')
                plt.plot(epoch_count, val_loss, 'b-')
                plt.legend(['Training Loss', 'Test Loss'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.show();
                '''
                
                prtstr += '\n'
                saver.save(sess, './models/'+savefilename)

                if FLAGS.debugmode=="1":
                    f = open('result.txt', 'a')
                    f.write(prtstr)
                    f.close()
                    f = open('result-short.txt', 'a')
                    f.write(prtstr2)
                    f.close()
            # arch view
            #writer.close()



    def get_batch_data(self, x, pos, sen_len, y, entity, batch_size, keep_prob0, keep_prob1, keep_prob2, eloc, is_shuffle=False):
        for index in batch_index(len(y), batch_size, 1, is_shuffle):
            feed_dict = {
                self.x: x[index],
                self.pos_x: pos[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.entity_id: entity[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
                self.keep_prob0: keep_prob0,
                #self.num_layers: num_layers,
                self.e_loc: eloc[index]
            }
            yield feed_dict, len(index)

def main(_):
    word_dict, w2v, p2id, p2v, maxlen = load_data_init(FLAGS.dataset, FLAGS.testset, FLAGS.embedding_file_path, FLAGS.embedding_file_path, FLAGS.embedding_dim)
    lstm = LSTM(
        len(word_dict),
        embedding_dim=FLAGS.embedding_dim,
        pos_embedding_dim = len(p2v),
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_class=FLAGS.n_class,
        max_sentence_len=maxlen,
        l2_reg=FLAGS.l2_reg,
        n_iter=FLAGS.n_iter,
        type_=FLAGS.method,
        num_layers=FLAGS.num_layers
    )
    lstm.run(PreEmbedding=w2v,word2id=word_dict,p2id=p2id,p2v=p2v)


if __name__ == '__main__':
    tf.app.run()
