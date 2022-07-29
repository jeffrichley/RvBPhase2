import os
import multiprocessing as mp
import logging
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from RvBSimulation import RvBSimulation
from RvBLearner import RvBLearner
from Memory import BasicMemory

# from pympler import tracker

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)
tf.get_logger().setLevel(logging.ERROR)


def run_one(parent_dir):

    # tr = tracker.SummaryTracker()

    print(parent_dir)
    start_load = time.time()
    memory = BasicMemory(memory_size=1)
    p1_learner = RvBLearner(num_joint_actions=9)
    p2_learner = RvBLearner(num_joint_actions=9)
    p1_learner.load_model(f'{parent_dir}/ddg_model')
    p2_learner.load_model(f'{parent_dir}/sag_model')
    config_file = './configs/phase2-config.ini'
    simulator = RvBSimulation(memory, p1_learner, p2_learner, 3, epsilon=0.99, epsilon_decay=0.9999, config_file=config_file)
    end_load = time.time()

    start_sim = time.time()
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

    end_sim = time.time()

    average = sum(local_history) / len(local_history)
    stdev = np.std(np.array(local_history))
    print(parent_dir, average, ddg_safe, sag_safe, all_safe, end_load - start_load, end_sim - start_sim, stdev)

    del simulator
    del p2_learner
    del p1_learner
    del memory

    # tr.print_diff()

    return average, ddg_safe, sag_safe, ddg_safe


if __name__ == '__main__':

    rewards = []
    steps = []

    rewards = [-2.35, 12.02, 115.38, 178.99, 57.63, 193.8, 484.17, 350.43, 228.77, 566.03, 75.93, 260.87, 459.07,
               400.36, 209.83, 395.23, 311.34, 464.09, 693.41, 11.09, 75.72, 436.96, 363.38, 23.61, 8.03, -50.4, 403.07,
               437.48, 398.68, 723.47, 792.62, 772.97, 679.61, 759.73, 840.23, 456.82, 846.3, 153.26, 642.83, 859.0,
               664.1, 821.37, 183.13, 676.17, 234.81, 828.72, 438.85, 460.02, 539.82, 666.86, 779.75, 733.23, 747.3,
               136.44, 859.49, 415.7, 747.77, 789.2, 687.91, 723.38, 829.33, 687.94, 837.83, 547.1, 822.9, 780.16,
               155.42, 464.71, 549.67, 887.89, 633.5, 680.29, 807.95, 595.34, 715.82, 454.75, 588.23, 931.38, 743.56,
               437.2, 468.03, 816.96, 302.79, 292.24, 640.13, 908.95, 956.73, 857.88, 891.26, 916.63, 924.63, 938.58,
               850.06, 706.1, 817.51, 795.31, 811.82, 970.22, 917.31, 775.46, 770.68, 717.74, 972.69, 825.79, 641.84,
               832.79, 777.29, 904.43, 734.32, 648.61, 883.3, 936.72, 906.37, 719.52, 858.9, 960.14, 909.89, 998.75,
               926.32, 913.83, 943.81, 988.58, 702.39, 949.45, 1008.58, 919.73, 949.91]
    steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000,
             9500, 10000, 10500, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000,
             17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500,
             25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000,
             32500, 33000, 33500, 34000, 34500, 35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500,
             40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000, 45500, 46000, 46500, 47000,
             47500, 48000, 48500, 49000, 50000, 50500, 51000, 51500, 52000, 52500, 53000, 53500, 54000, 54500, 55500,
             56000, 56500, 57000, 57500, 58000, 58500, 61000, 61500, 62000, 62500, 63000, 63500, 64000, 64500, 66500,
             67000, 67500, 68000, 68500]

    print(len(steps))

    # model_dir = './models'
    model_dir = './experiments/2.5/models'
    # model_dir = 'D:/workspaces/work/RvBPhase2/experiments/2.4/models'
    start_model = 500

    model = start_model
    model_dirs = []
    while os.path.exists(f'{model_dir}/{model}'):
        current_model_dir = f'{model_dir}/{model}'
        if model not in steps:
            # avg = run_one(current_model_dir)
            # rewards.append(avg)
            # steps.append(model)
            # print(model, avg)
            # print(rewards)
            # print(steps)
            print('adding', model)
            model_dirs.append(current_model_dir)
        model += 500

    print(f'processing {len(model_dirs)} models')

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(run_one, model_dirs)
    pool.close()

    print('final')
    print(results)

