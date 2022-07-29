import multiprocessing as mp
import os
import logging
import threading
from functools import partial

import tensorflow as tf

from RvBSimulation import RvBSimulation
from RvBLearner import RvBLearner
from Memory import BasicMemory



def run_one(parent_dir):
    # parent_dir, num_runs = vals
    print('start run_one', parent_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('WARNING')
    tf.autograph.set_verbosity(1)
    tf.get_logger().setLevel(logging.ERROR)

    memory = BasicMemory(memory_size=100000)
    p1_learner = RvBLearner(num_joint_actions=9)
    p2_learner = RvBLearner(num_joint_actions=9)
    p1_learner.load_model(f'{parent_dir}/ddg_model')
    p2_learner.load_model(f'{parent_dir}/sag_model')
    config_file = './configs/phase2-config.ini'
    simulator = RvBSimulation(memory, p1_learner, p2_learner, 3, epsilon=0.99, epsilon_decay=0.9999, config_file=config_file)

    local_history = []
    ddg_safe = 0
    sag_safe = 0
    all_safe = 0
    for i in range(100):
        test_reward, test_number_of_steps = simulator.run_game_iteration(use_epsilon_greedy=False,
                                                                         render_video=False,
                                                                         learn=False)
        local_history.append(test_reward)
        if simulator.previous_ddg_safe:
            ddg_safe += 1
            if simulator.previous_sag_safe:
                all_safe += 1

        if simulator.previous_sag_safe:
            sag_safe += 1

    average = sum(local_history) / len(local_history)
    return average, ddg_safe, sag_safe, all_safe


class NetworkEvaluator:

    def __init__(self, log_file, num_runs=100):

        self.tb_logger = tf.summary.create_file_writer(log_file)
        scores = tf.keras.metrics.Mean('100_test_avg_score', dtype=tf.float32)
        ddg_safe = tf.keras.metrics.Mean('100_test_num_ddg_safe', dtype=tf.int32)
        sag_safe = tf.keras.metrics.Mean('100_test_num_sag_safe', dtype=tf.int32)
        all_safe = tf.keras.metrics.Mean('100_test_num_all_safe', dtype=tf.int32)

        self.score_history = []
        self.pool = mp.Pool(mp.cpu_count() // 2)
        self.results = None
        self.num_runs = num_runs

    def evaluate(self, model_dir):
        x = threading.Thread(target=self.start_eval, args=(model_dir,))
        x.start()

    def start_eval(self, model_dir):
        print('start_eval')
        answers = self.pool.map(run_one, (model_dir,))
        # answers = self.results.get()
        # answers = mp.Process(target=run_one, args=(model_dir, self.num_runs)).get()
        score, ddg_safe, sag_safe, all_safe = answers

        with self.tb_logger.as_default():
            tf.summary.scalar('100_test_avg_score', score, step=1)
            tf.summary.scalar('100_test_num_ddg_safe', ddg_safe, step=1)
            tf.summary.scalar('100_test_num_sag_safe', sag_safe, step=1)
            tf.summary.scalar('100_test_num_all_safe', all_safe, step=1)
            # scores = tf.keras.metrics.Mean('100_test_avg_score', dtype=tf.float32)
            # ddg_safe = tf.keras.metrics.Mean('100_test_num_ddg_safe', dtype=tf.int32)
            # sag_safe = tf.keras.metrics.Mean('100_test_num_sag_safe', dtype=tf.int32)
            # all_safe = tf.keras.metrics.Mean('100_test_num_all_safe', dtype=tf.int32)

        print(answers)

    def finished(self):
        print(self.results.get())
