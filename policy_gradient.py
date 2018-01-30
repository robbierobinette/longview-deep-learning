import random
import collections
import numpy as np
from array import array
import time
from timings import Timings
import tensorflow as tf
from tf_helpers import *


class QModelBase(object):
    def __init__(self):
        self.state_size = None
        self.n_actions = None
        self.output_size = None
        self.label = "none"
        self.session: tf.Session = None
        self.keep_prob = None
        self.x = None
        self.action = None
        self.mask = None
        self.grad_update = None
        self.tensorboard_dir = 'tf-logs'

    def add_summaries(self):
        self.score = tf.placeholder(tf.float32, shape=[])
        self.score_summary = tf.summary.scalar("score-10k", self.score)
        self.merged_summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.tensorboard_dir + "/" + self.label, self.session.graph)

    def add_score(self, score: float, step: int):
        ss = self.session.run([self.merged_summaries], feed_dict = {self.score: score})
        self.writer.add_summary(ss[0], step)

    def close_session(self):
        self.session.close()

    def get_action(self, state: np.ndarray) -> (int, np.ndarray):
        feed_dict = {self.x: state}
        if self.keep_prob is not None:
            feed_dict[self.keep_prob] = 1
        action = self.session.run(fetches = [self.action], feed_dict = feed_dict)
        return (action[0], [])

    def train_one_batch(self, state: np.ndarray, mask: np.ndarray, rewards: np.ndarray):
        feed_dict = { self.x : state, self.mask: mask, self.y : rewards}
        if self.keep_prob is not None:
            feed_dict[self.keep_prob] = .5
        self.session.run(self.grad_update, feed_dict = feed_dict)

class QModel(QModelBase):

    def __init__(self,
                 session: tf.Session,
                 label: str,
                 n_actions: int,
                 state_size: int,
                 layer_size: int,
                 dropout: bool,
                 res_blocks: int,
                 res_layers: int,
                 layers: int,
                 layer_norm: bool,
                 learn_rate: float,
                 tensorboard_dir):

        super().__init__()

        state_frames = 1
        self.label = label
        self.session = session
        self.tensorboard_dir = tensorboard_dir

        input_size = state_size * state_frames
        print("input_size %d" % input_size)
        x = tf.placeholder(tf.float32, [None, input_size], name = "X")
        tip = x

        if (dropout):
            keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
        else:
            keep_prob = None

        for i in range(res_blocks):
            tip = add_residual_block(tip, layer_size, res_layers, layer_norm = layer_norm, keep_prob = keep_prob, name = ("res-block-%d" % i))

        if res_blocks == 0:
            for i in range(0, layers):
                #tip = add_dense(tip, layer_size, name = ("layer-%d" % i))
                tip = add_dense_norm(tip, layer_size, layer_norm = layer_norm, keep_prob = keep_prob,  name = ("layer-%d" % i))

        output_size = n_actions
        action_scores = tf.identity(tf.layers.dense(tip, output_size), name = "action_scores")

        action = tf.argmax(action_scores, axis=1, name = "action")

        print("output_layer.size: ", action_scores.shape)

        y = tf.placeholder(tf.float32, [None, output_size], name = "Y")
        error_mask = tf.placeholder(tf.float32, [None, output_size], name = "Y_mask")
        error = tf.multiply(tf.square(action_scores - y), error_mask)

        network_params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, use_locking = False)
        grad_update = optimizer.minimize(error, var_list = network_params, name = "grad_update")

        self.x = x
        self.output_layer = action_scores
        self.output_size = output_size
        self.action = action
        self.n_actions = n_actions
        self.input_size = input_size
        self.y = y
        self.mask = error_mask
        self.error = error
        self.network_params = network_params
        self.optimizer = optimizer
        self.grad_update = grad_update
        self.state_frames = state_frames
        self.state_size = state_size
        self.keep_prob = keep_prob

        self.add_summaries()
        self.session.run(tf.global_variables_initializer())


class QModelFromSavedFile(QModelBase):
    def __init__(self, session, path):
        super().__init__()
        saver = tf.train.import_meta_graph(path + ".meta")
        saver.restore(session, path)
        g = tf.get_default_graph()
        self.x = g.get_tensor_by_name("X:0")
        self.action_scores = g.get_tensor_by_name("action_scores:0")
        self.action = g.get_tensor_by_name("action:0")
        self.mask = g.get_tensor_by_name("Y_mask:0")
        self.keep_prob = g.get_tensor_by_name("keep_prob:0")

        self.state_size = int(self.x.shape[1])
        self.n_actions = self.action_scores.shape[1]
        self.output_size = self.action_scores.shape[1]
        print("state_size %d n_actions %d output_size %d" % (self.state_size, self.n_actions, self.output_size))
        self.label = "none"
        self.session: tf.Session = session
        self.grad_update = g.get_operation_by_name("grad_update")
        self.tensorboard_dir = 'tf-logs'


class StateActionReward(object) :
    def __init__(self, state: np.ndarray, action: int, reward: float):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = None

    def add_reward(self, reward) :
        self.reward += reward

    def set_next_state(self, next_state) :
        self.next_state = next_state

    def training_sample(self, model: QModel) -> (np.ndarray, np.ndarray) :
        state_size = model.state_size
        n_actions = model.n_actions

        if (model.output_size == model.n_actions):
            mask = np.zeros(shape = [1, n_actions ])
            output = np.zeros(shape = [1, n_actions])
            mask[0, self.action] = 1
            output[0, self.action] = self.reward
            return self.state, mask, output
        else:
            mask = np.zeros(shape = [1, n_actions + state_size * n_actions])
            output = np.zeros(shape = [1, n_actions + state_size * n_actions])
            mask[0, self.action] = 1
            output[0, self.action] = self.reward

            state_start = self.action * state_size + n_actions
            state_end = state_start + state_size

            output[0, state_start : state_end] = self.next_state
            mask[0, state_start : state_end] = 1

            return self.state, mask, output


class PolicyGradientOptimizer(object) :
    def __init__(self,
                 model: QModel,
                 input_size,
                 n_actions,
                 max_memory_size: int,
                 action_buffer_size: int,
                 gamma: float,
                 exploration: float,
                 timings: Timings) :
        self.model = model
        self.state_size = model.state_size
        self.input_size = input_size
        self.n_actions = n_actions
        self.max_memory_size = max_memory_size
        self.action_buffer_size = action_buffer_size
        self.gamma = gamma
        self.action_buffer = collections.deque()
        self.memory = []
        self.exploration = exploration
        self.timings = timings

    def memory_size(self) -> int :
        return len(self.memory)

    def ready(self) -> bool :
        return (self.memory_size() > 10240)

    def add_action(self, state: np.ndarray, action: int, reward: float) :

        self.action_buffer.appendleft(StateActionReward(state, action, reward))
        if (reward != 0.0) :
            for sar in self.action_buffer :
                reward *= self.gamma
                sar.add_reward(reward)

        # add buffer events
        while len(self.action_buffer) > self.action_buffer_size :
            sar = self.action_buffer.pop()
            sar.set_next_state(self.action_buffer[-1].state)
            self.add_to_memory(sar)

        # trim memory
        while len(self.memory) > self.max_memory_size :
            self.memory.pop()

    # if the memory is not yet full append to the existing memory.  If it is full
    # then replace a randomly chosen sample.
    def add_to_memory(self, sar):
        if len(self.memory) < self.max_memory_size:
            self.memory.append(sar)
        else:
            idx = np.random.randint(0, len(self.memory))
            self.memory[idx] = sar

    # this is for when we have ended an episode and the new set of actions and rewards are independent
    # of the the previous actions.  For instance, a new ball in pong.
    def flush_action_buffer(self):

        n = len(self.action_buffer)

        while len(self.action_buffer) > 0:
            sar = self.action_buffer.pop()
            if (len(self.action_buffer) > 0):
                sar.set_next_state(self.action_buffer[-1].state)
            self.add_to_memory(sar)

        self.action_buffer = collections.deque()
        return n

    def train(self, n_epochs: int, batch_size: int) :
        for i in range(n_epochs) :
            self.train_one_batch(batch_size)

    def get_action(self, input: np.ndarray) -> (int, np.ndarray) :
        a, state = self.model.get_action(input)
        return a, state

    def train_one_batch(self, batch_size: int) :

        if not self.ready() :
            return

        t = time.time()
        input_rows = []
        output_rows = []
        mask_rows = []

        for i in range(batch_size) :
            idx = np.random.randint(0, len(self.memory))
            sar = self.memory[idx]
            irow, mask, orow = sar.training_sample(self.model)
            input_rows.append(irow)
            mask_rows.append(mask)
            output_rows.append(orow)

        self.timings.add('assemble sar', time.time() - t)
        t = time.time()
        input_block = np.vstack(input_rows)
        output_block = np.vstack(output_rows)
        mask_block = np.vstack(mask_rows)
        self.timings.add('vstack', time.time() - t)

        self.model.train_one_batch(input_block, mask_block, output_block)
        self.timings.add('train_one_batch', time.time() - t)
