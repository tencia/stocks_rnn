from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas_datareader.data as pdr_data
import numpy as np
import time
import os
import sys
from collections import deque

import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import config as c

"""
Adapted from Google's PTB word prediction TensorFlow tutorial.

Copyright 2016 Tencia Lee

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def get_data():
    '''
    If filename exists, loads data, otherwise downloads and saves data
    from Yahoo Finance
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def download_data():
        from datetime import timedelta, datetime
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        print('Downloading data for dates {} - {}'.format(
            datetime.strftime(c.start, "%Y-%m-%d"),
            datetime.strftime(c.end, "%Y-%m-%d")))
        split = [0.8, 0.1, 0.1]
        cumusplit = [np.sum(split[:i]) for i,s in enumerate(split)]
        segment_start_dates = [c.start + timedelta(
            days = int((c.end - c.start).days * interv)) for interv in cumusplit][::-1]
        stocks_list = map(lambda l: l.strip(), open(c.names_file, 'r').readlines())
        by_stock = dict((s, pdr_data.DataReader(s, 'yahoo', c.start, c.end))
                for s in stocks_list)
        seq = [[],[],[]]
        for stock in by_stock:
            lastAc = -1
            daily_returns = deque(maxlen=c.normalize_std_len)
            for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end-c.start).days)):
                idx = next(i for i,d in enumerate(segment_start_dates) if rec_date >= d)
                try:
                    d = rec_date.strftime("%Y-%m-%d")
                    ac = by_stock[stock].ix[d]['Adj Close']
                    daily_return = (ac - lastAc)/lastAc
                    if len(daily_returns) == daily_returns.maxlen:
                        seq[idx].append(daily_return/np.std(daily_returns))
                    daily_returns.append(daily_return)
                    lastAc = ac
                except KeyError:
                    pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq][::-1]
    
    if not os.path.exists(c.save_file):
        datasets = download_data()
        print('Saving in {}'.format(c.save_file))
        np.savez(c.save_file, *datasets)
    else:
        with np.load(c.save_file) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets

def seq_iterator(raw_data, batch_size, num_steps):
    """
    Iterate on the raw return sequence data.
    Args:
    - raw_data: array
    - batch_size: int, the batch size.
    - num_steps: int, the number of unrolls.
    Yields:
    - Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
    - ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.float32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.float32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y) 

class StockLSTM(object):
    """
    This model predicts a 1D sequence of real numbers (here representing daily stock adjusted
    returns normalized by running fixed-length standard deviation) using an LSTM.
    It is regularized using the method in [Zaremba et al 2015]
    http://arxiv.org/pdf/1409.2329v5.pdf
    """
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)]
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])

        self._output = output = tf.nn.xw_plus_b(rnn_output,
                                 tf.get_variable("out_w", [size, 1]),
                                 tf.get_variable("out_b", [1]))

        self._cost = cost = tf.reduce_mean(tf.square(output - tf.reshape(self._targets, [-1])))
        self._final_state = states[-1]

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost
    
    @property
    def output(self):
        return self._output

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
    

def main(config_size='small', num_epochs=10):
    
    def get_config(config_size):
        config_size = config_size.lower()
        if config_size == 'small':
            return c.SmallConfig()
        elif config_size == 'medium':
            return c.MediumConfig()
        elif config_size == 'large':
            return c.LargeConfig()
        else:
            raise ValueError('Unknown config size {} (small, medium, large)'.format(config_size))

    def run_epoch(session, m, data, eval_op, verbose=False):
        """Runs the model on the given data."""
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        print(epoch_size)
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = m.initial_state.eval()
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                             {m.input_data: x, m.targets: y, m.initial_state: state})
            costs += cost
            iters += m.num_steps

            print_interval = 20
            if verbose and epoch_size > print_interval \
                    and step % (epoch_size // print_interval) == print_interval:
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs / iters,
                     iters * m.batch_size / (time.time() - start_time)))
        return costs / (iters if iters > 0 else 1)

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config(config_size)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = StockLSTM(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = StockLSTM(is_training=False, config=config)

        tf.initialize_all_variables().run()

        train_data, valid_data, test_data = get_data()

        for epoch in xrange(num_epochs):
            lr_decay = config.lr_decay ** max(epoch - num_epochs, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            cur_lr = session.run(m.lr)

            mse = run_epoch(session, m, train_data, m.train_op, verbose=True)
            vmse = run_epoch(session, mtest, valid_data, tf.no_op())
            print("Epoch: %d - learning rate: %.3f - train mse: %.3f - test mse: %.3f" %
                  (epoch, cur_lr, mse, vmse))

        tmse = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test mse: %.3f" % tmse)

if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name,arg_type in zip(ma.args[-len(ma.defaults):],[type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
