import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Ensure TensorFlow 1.x compatibility
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dqn_batch import DQNPrioritizedReplay
from uavs_env import UAVSEnv
import os
import random
from multiprocessing import Pool, cpu_count

# Parameters
TARGET = np.array([9, 9, 4])

n_actions=50
n_features=2
n_embedding=32
learning_rate=0.0005
reward_decay=0.5
e_greedy=0.9
e_greedy_increment=0.00012
replace_target_iter=1000
memory_size=5000
batch_size=32
#e_greedy_increment=None
#e_greedy_increment=0.00015
prioritized=True
output_graph=True

class SumTree:
    """SumTree for Prioritized Experience Replay."""
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory:
    """Memory for Prioritized Experience Replay."""
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size), dtype=object), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQNPrioritizedReplay:
    """DQN with Prioritized Experience Replay."""
    def __init__(
            self,
            n_actions,
            n_features,
            n_embedding,
            learning_rate=0.0005,
            reward_decay=0.5,
            e_greedy=0.9,
            replace_target_iter=1000,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_embedding = n_embedding
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.method = 'pri_dqn'
            self.memory = Memory(capacity=memory_size)
        else:
            self.method = 'dqn'
            self.memory = np.zeros((self.memory_size, 4), dtype=object)

        if sess is None:
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, adj, mean_matrix, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.compat.v1.variable_scope('l_emb'):
                w_emb = tf.compat.v1.get_variable('w_emb', [self.n_features, self.n_embedding], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b_emb = tf.compat.v1.get_variable('b_emb', [1, self.n_embedding], initializer=b_initializer, collections=c_names,  trainable=trainable)
                output = tf.matmul(s, w_emb)
                embedding_s = tf.nn.relu(tf.matmul(adj, output) + b_emb)
                embedding_avg_s = tf.matmul(mean_matrix, embedding_s)

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_embedding, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(embedding_avg_s, w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input_state_feature
        self.adj = tf.compat.v1.placeholder(tf.float32, [None, None], name='adj')  # input_adj_matrix
        self.mean_matrix = tf.compat.v1.placeholder(tf.float32, [None, None], name='mean_m')  # input_adj_matrix
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.compat.v1.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, self.adj, self.mean_matrix, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.compat.v1.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.compat.v1.losses.mean_squared_error(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(self.q_target, self.q_eval))
        with tf.compat.v1.variable_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        self.adj_ = tf.compat.v1.placeholder(tf.float32, [None, None], name='adj_')  # input_adj_matrix
        self.mean_matrix_ = tf.compat.v1.placeholder(tf.float32, [None, None], name='mean_m_')  # input_adj_matrix
        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.adj_, self.mean_matrix_, c_names, n_l1, w_initializer, b_initializer, False)

    def store_transition(self, s, a, r, s_):
        transition = np.array([s, a, r, s_], dtype=object)
        if self.prioritized:
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def laplacian_matrix_sys_normalized(self, s):
        adj_matrix = np.array(nx.adjacency_matrix(s).todense())
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
        exist = (adj_matrix > 0) * 1
        factor = np.ones(adj_matrix.shape[1])
        degree = np.dot(exist, factor)
        d_hat = np.diag(np.power(degree, -0.5).flatten())
        norm_adj = d_hat.dot(adj_matrix).dot(d_hat)
        return norm_adj, degree

    def choose_action(self, observation, steps):
        graph = observation.copy()
        remain_node = graph.nodes()
        adj, degree = self.laplacian_matrix_sys_normalized(graph)
        state_feature_w = np.transpose(np.matrix(list(nx.get_node_attributes(graph, 'weight').values())))
        state_feature_d = np.transpose((np.matrix(degree)) / (max(degree)))
        state_feature = np.hstack((state_feature_w, state_feature_d))
        mean_matrix = np.ones((1, len(remain_node))) / len(remain_node)
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state_feature, self.adj: adj, self.mean_matrix: mean_matrix})
            for node in steps:
                actions_value[0][node] = -inf
            action = np.argmax(actions_value)
        else:
            action = random.choice(list(remain_node.keys()))
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        batch_s = batch_memory[:, 0]
        batch_s_ = batch_memory[:, 3]

        ba_s = Batchgraph(self.batch_size, batch_s)
        batched_adj, batched_feature, mean_matrix = ba_s.batched_graph()
        ba_s_ = Batchgraph(self.batch_size, batch_s_)
        batched_adj_, batched_feature_, mean_matrix_ = ba_s_.batched_graph()

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batched_feature_, self.adj_: batched_adj_, self.mean_matrix_: mean_matrix_,
                       self.s: batched_feature, self.adj: batched_adj, self.mean_matrix: mean_matrix})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, 1].astype(int)
        reward = batch_memory[:, 2].astype(float)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            opp, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                       feed_dict={self.s: batched_feature,
                                                                  self.adj: batched_adj,
                                                                  self.mean_matrix: mean_matrix,
                                                                  self.q_target: q_target,
                                                                  self.ISWeights: ISWeights})
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batched_feature,
                                                    self.adj: batched_adj,
                                                    self.mean_matrix: mean_matrix,
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        if self.prioritized:
            self.memory.batch_update(tree_idx, abs_errors)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        np.savetxt("cost_his.txt", self.cost_his)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


class UAVSEnv:
    # Placeholder for the UAV environment.
    def state0(self, G):
        return G  # Placeholder, implement accordingly.