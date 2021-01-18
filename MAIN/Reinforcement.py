import tensorflow as tf
import PROCESSOR.MachineLearning as ML
from MAIN.Basics import Agent
import random
random.seed(0)

class ContextualBandit(Agent):

    def __init__(self, network, config, reward_engine):
        super().__init__(network, config)
        self.epoch_counter = self.counters[config['AgentEpochCounter']]
        self.iter_counter  = self.counters[config['AgentIterationCounter']]

        # Placeholder
        self.temp = tf.placeholder(shape=[1], dtype=tf.float32)
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

        # Algorithm
        self.output    = tf.reshape(self.output_layer, [-1])
        self.prob_dist = tf.nn.softmax(self.output / self.temp)
        self.weight    = tf.slice(self.output, self.action_holder, [1])
        self.loss      = -(tf.log(self.weight) * self.reward_holder)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config['AgentLearningRate'])
        self.update    = self.optimizer.minimize(self.loss)

        # Processor
        self.exploration   = ML.Exploration(self)
        self.exp_buffer    = ML.ExperienceBuffer(self)
        self.state_space   = ML.StateSpace(self)
        self.action_space  = ML.ActionSpace(self)
        self.reward_engine = ML.RewardEngine(self, reward_engine)
        self.recorder      = ML.Recorder(self)

        # Record action evolution with steps
        self.net_act_evolution = []

    def update_network(self):
        if self.config['AgentIsUpdateNetwork'] is True:
            if (self.iter_counter.is_buffered is True) or (self.iter_counter.n_buffer == 0):
                if (self.config['ExperienceReplay'] is True) \
                        and (len(self.exp_buffer.buffer) > 0) \
                        and (self.iter_counter.value % self.config['ExperienceReplayFreq'] == 0):
                    self.exp_buffer.process('get')
                    state  = self.data['EXPERIENCE_BUFFER_SAMPLE'][0]
                    action = self.data['EXPERIENCE_BUFFER_SAMPLE'][1]
                    reward = self.data['EXPERIENCE_BUFFER_SAMPLE'][2]
                else:
                    state  = self.data['NETWORK_STATE']
                    action = self.data['NETWORK_ACTION']
                    reward = self.data['ENGINE_REWARD']

                self.feed_dict[self.input_layer  ] = [int(state)]
                self.feed_dict[self.action_holder] = [int(action)]
                self.feed_dict[self.reward_holder] = [reward]

                _ = self.session.run([self.update], feed_dict=self.feed_dict)

    def buffering(self):
        if self.config['ExperienceReplay'] is True:
            self.create_sample_list()
            self.exp_buffer.process('add')

    def create_sample_list(self):
        state  = self.data['NETWORK_STATE']
        action = self.data['NETWORK_ACTION']
        reward = self.data['ENGINE_REWARD']
        self.data['SAMPLE'] = [[state, action, reward]]

    def process(self, session, save=False, restore=False, train=True):
        self.set_session(session)
        self.initialize_global()
        if restore is True: self.restore_model()

        if train:
            while self.epoch_counter.is_ended is False:
                net_act_evolution_iter = []
                while self.iter_counter.is_ended is False:
                    index = random.randint(601, len(self.reward_engine.engine.x) - 1201)

                    curr_pri_diff = self.reward_engine.engine.y[index] - self.reward_engine.engine.x[index]
                    hist_pri_diff = self.reward_engine.engine.hist_pri_diff
                    curr_env_state = (curr_pri_diff/hist_pri_diff) - 1.
                    curr_env_state = min(0.1, max(-0.1, curr_env_state))
                    #curr_env_state = 1.1 - 1.
                    self.state_space.process(method='binary', env_state=curr_env_state)

                    self.action_space.process()
                    self.reward_engine.process(index)
                    self.buffering()
                    self.update_network()
                    self.recorder.process()
                    self.iter_counter.step()                
                    #net_act_evolution_iter.append(self.data['NETWORK_ACTION']) 
                    net_act_evolution_iter.append(self.action_space.convert(self.data['NETWORK_ACTION'],'index_to_dict')) 
                    # Add a convergence check.

                self.iter_counter.reset()
                self.epoch_counter.step()
                self.net_act_evolution.append(net_act_evolution_iter)

            self.epoch_counter.reset()

            if save is True: self.save_model()
