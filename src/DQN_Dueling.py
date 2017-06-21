import tensorflow as tf
import numpy as np
import time
import os


def clear():
    os.system('cls')


# Hyperparameters
OBSERVE = 100000.0
EXPLORE = 2000000.0
BATCH_SIZE = 32
GREEDY_POLICY_EPSILON = 0.9
EPSILON_INITIAL = 0.9
EPSILON_MAX = 0.9999
EPSILON_INCREAMENT = 0.0000001
DISCOUNT_FACTOR_GAMMA = 0.9
MAX_MEMORY_CAPACITY = 5000
MEMORY_COUNTER = 0
LEARNING_STEP_COUNTER = 0
LEARNING_RATE = 0.01
NUMBER_OF_ACTIONS = 3
NUMBER_OF_STATES = 2
TARGET_UPDATE_INTERVAL = 10000

# Initializing Memory
MEMORYS = np.zeros((MAX_MEMORY_CAPACITY, NUMBER_OF_STATES, 80, 80))
MEMORYA = np.zeros((MAX_MEMORY_CAPACITY, 1), dtype=int)
MEMORYR = np.zeros((MAX_MEMORY_CAPACITY, 1))
MEMORYS_ = np.zeros((MAX_MEMORY_CAPACITY, NUMBER_OF_STATES, 80, 80))


class DeepQNetwork:
    def __init__(self):
        # tf placeholders
        self.current_state = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_STATES, 80, 80], name='CUR_STATE')
        self.next_state = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_STATES, 80, 80], name='NEXT_STATE')
        self.action = tf.placeholder(tf.int32, shape=[None, 1], name='ACTIONS')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='REWARD')
        # self.input_layer_ = tf.transpose(self.next_state, perm=[0, 2, 3, 1])

        with tf.variable_scope('Dueling_DQN'):
            self.__build_net()

            # Initialize Setup
            self.saver = tf.train.Saver()
            self.logwriter = tf.summary.FileWriter('/board')
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # Logging
            self.logwriter.add_graph(self.sess.graph)

    def __build_net(self):
        def build_cnn(frame_stack):
            with tf.variable_scope('CNN'):
                # Convolutional Layer #1
                convolutional_layer_1 = tf.layers.conv2d(
                    inputs=frame_stack,  # inputs --> last layer
                    strides=4,
                    filters=32,  # filters --> number of output layers OR new depth
                    kernel_size=[8, 8],  # kernel_size --> size of each scanning area
                    padding="SAME",  # padding --> edge processing method
                    activation=tf.nn.relu
                )

                # Pooling Layer #1
                pooling_layer_1 = tf.layers.max_pooling2d(
                    inputs=convolutional_layer_1,  # inputs --> last layer
                    pool_size=2,
                    strides=2,
                    padding="SAME",  # padding --> edge processing method
                )

                # Convolutional Layer #2
                convolutional_layer_2 = tf.layers.conv2d(
                    inputs=pooling_layer_1,  # inputs --> last layer
                    strides=2,
                    filters=64,  # filters --> number of output layers OR new depth
                    kernel_size=[4, 4],  # kernel_size --> size of each scanning area
                    padding="SAME",  # padding --> edge processing method
                    activation=tf.nn.relu
                )

                # Pooling Layer #2
                pooling_layer_2 = tf.layers.max_pooling2d(
                    inputs=convolutional_layer_2,  # inputs --> last layer
                    pool_size=2,
                    strides=2,
                    padding="SAME",  # padding --> edge processing method
                )

                # Convolutional Layer #3
                convolutional_layer_3 = tf.layers.conv2d(
                    inputs=pooling_layer_2,  # inputs --> last layer
                    strides=1,
                    filters=64,  # filters --> number of output layers OR new depth
                    kernel_size=[3, 3],  # kernel_size --> size of each scanning area
                    padding="SAME",  # padding --> edge processing method
                    activation=tf.nn.relu
                )

                # Pooling Layer #3
                pooling_layer_3 = tf.layers.max_pooling2d(
                    inputs=convolutional_layer_3,  # inputs --> last layer
                    pool_size=2,
                    strides=2,
                    padding="SAME",  # padding --> edge processing method
                )

                return tf.reshape(pooling_layer_3, [-1, 128])

        def build_dqn(state, w, b):
            with tf.variable_scope('Dense'):
                ly1 = tf.layers.dense(
                    inputs=state,
                    units=64,
                    activation=tf.nn.relu,
                    kernel_initializer=w,
                    bias_initializer=b
                )

            with tf.variable_scope('Value'):
                val = tf.layers.dense(
                    inputs=ly1,
                    units=1,
                    activation=None,
                    kernel_initializer=w,
                    bias_initializer=b
                )

            with tf.variable_scope('Advantage'):
                adv = tf.layers.dense(
                    inputs=ly1,
                    units=NUMBER_OF_ACTIONS,
                    activation=None,
                    kernel_initializer=w,
                    bias_initializer=b
                )

            with tf.variable_scope('Q'):
                out = val + (adv - tf.reduce_mean(adv, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)

            return out

        # ------------------ build evaluate_net ------------------
        self.q_target = tf.placeholder(tf.float32, [None, NUMBER_OF_ACTIONS], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            w_initializer = tf.random_normal_initializer(0.0, 0.3)
            b_initializer = tf.constant_initializer(0.1)
            current_state_processed = build_cnn(self.current_state)
            self.q_evaluation = build_dqn(current_state_processed, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_evaluation))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            next_state_processed = build_cnn(self.next_state)
            self.q_next = build_dqn(next_state_processed, w_initializer, b_initializer)

    def __choose_action(self, state):
        # current_state = state[np.newaxis, :]
        if np.random.uniform() < GREEDY_POLICY_EPSILON:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_evaluation, feed_dict={self.current_state: [state]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, NUMBER_OF_ACTIONS)
        return action

    @staticmethod
    def __store_transition(s, a, r, s_):
        global MEMORY_COUNTER
        # replace the old memory with new memory
        index = MEMORY_COUNTER % MAX_MEMORY_CAPACITY
        MEMORYS[index] = s
        MEMORYA[index] = a
        MEMORYR[index] = r
        MEMORYS_[index] = s_
        MEMORY_COUNTER += 1

    def __replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def __learn(self, step):
        global GREEDY_POLICY_EPSILON, EPSILON_MAX, LEARNING_STEP_COUNTER
        if LEARNING_STEP_COUNTER % TARGET_UPDATE_INTERVAL == 0:
            self.__replace_target_params()
            print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(MAX_MEMORY_CAPACITY, BATCH_SIZE)
        b_s = MEMORYS[sample_index]
        b_a = MEMORYA[sample_index].astype(int)
        b_r = MEMORYR[sample_index]
        b_s_ = MEMORYS_[sample_index]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_evaluation],
            feed_dict={self.next_state: b_s_,  # next observation
                       self.current_state: b_s_}  # next observation
        )

        q_eval = self.sess.run(self.q_evaluation, {self.current_state: b_s})

        q_target = q_eval.copy()

        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)

        max_act4next = np.argmax(q_eval4next, axis=1)

        selected_q_next = q_next[batch_index, max_act4next]

        q_target[batch_index, b_a] = b_r + DISCOUNT_FACTOR_GAMMA * selected_q_next

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.current_state: b_s,
                                                self.q_target: q_target})

        if GREEDY_POLICY_EPSILON < EPSILON_MAX and step > OBSERVE:
            GREEDY_POLICY_EPSILON += EPSILON_INCREAMENT
        else:
            GREEDY_POLICY_EPSILON = EPSILON_MAX
        LEARNING_STEP_COUNTER += 1

    def __save(self, e):
        self.saver.save(self.sess, 'saved_networks/breakout', global_step=e)

    def start_playing(self, env):
        global EPSILON_INCREAMENT, GREEDY_POLICY_EPSILON, MAX_MEMORY_CAPACITY
        print('\nCollecting experience...')

        # Try loading Logs
        try:
            fin = open('logs/checkpoint', 'r')

        except IOError:
            i_episode = 0  # might need to activate manually
            step = 1
            skipped_step = 0
            print('No history logs found.')

        else:
            line = fin.read().split(' ')
            fin.close()
            i_episode = int(line[0])
            step = int(line[1])
            skipped_step = step
            GREEDY_POLICY_EPSILON += (step + MAX_MEMORY_CAPACITY) * EPSILON_INCREAMENT

        if step > 0:
            checkpoint = tf.train.get_checkpoint_state("saved_networks/breakout")
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        scoreout = None

        total_time = 0

        while True:
            if i_episode % 1000 == 0:
                scoreout = open('logs/score{0}_{1}'.format(i_episode + 1, i_episode + 1000), 'w')
            curr_state = env.restart()
            episode_reward = 0

            start_time = time.time()

            while True:
                action = self.__choose_action(curr_state)
                next_state = np.zeros((2, 80, 80), dtype=float)
                rewards = np.array([0.0, 0.0], dtype=float)

                # take action
                next_state[0], rewards[0], is_game_over = env.render(action)
                if not is_game_over:
                    next_state[1], rewards[1], is_game_over = env.render(action)
                else:
                    next_state[1], rewards[1] = next_state[0], 0

                """
                if not is_game_over:
                    next_state[2], rewards[2], is_game_over = env.render(action)
                else:
                    next_state[2], rewards[2] = next_state[1], 0
                if not is_game_over:
                    next_state[3], rewards[3], is_game_over = env.render(action)
                else:
                    next_state[3], rewards[3] = next_state[2], 0
                """

                reward = np.sum(rewards).item(0)

                self.__store_transition(curr_state, action, reward, next_state)

                episode_reward += reward

                if MEMORY_COUNTER > MAX_MEMORY_CAPACITY:
                    self.__learn(step)
                    if is_game_over and scoreout is not None:
                        scoreout.write(
                            '{0} {1} {2}\n'.format(i_episode, episode_reward, env.seconds + env.ticks / env.fps))

                print(
                    'Episode:', i_episode,
                    ' | Step: ', step,
                    ' | Step Rewards: {0:5.2f}'.format(reward),
                    ' | Epsilon: {0:.6f}'.format(GREEDY_POLICY_EPSILON),
                    ' | Avg Survial Time: {0:.6f}', total_time / (step - skipped_step), ' s'
                )

                step += 1

                if step % 10000 == 0 and step >= 50000:
                    self.__save(step)
                    fout = open('logs/checkpoint', 'w')
                    fout.write(str(i_episode) + ' ' + str(step))
                    fout.close()

                if is_game_over:
                    break
                curr_state = next_state

            episode_time = time.time() - start_time
            total_time += episode_time
            i_episode += 1

            if i_episode % 1000 == 0 and scoreout is not None:
                scoreout.close()
