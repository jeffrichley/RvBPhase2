import time
import os
import datetime

import numpy as np
import tensorflow as tf

from RvBSimulation import RvBSimulation
from Memory import BasicMemory
from RvBLearner import RvBLearner

from utils import get_stacked_coco_values, get_queryable_state_info


def run():
    number_individual_actions = 4
    number_joint_actions = number_individual_actions * number_individual_actions

    # learning configs
    batch_size = 1024  # worked
    # batch_size = 128
    gamma = 0.99
    epsilon = 0.99
    # epsilon = 0.3464365528128901
    epsilon_decay = 0.99999
    # epsilon_decay = 1.0
    update_target_network = 15000  # TODO: need to calculate a better number based on episodes

    all_main_time = []
    all_simulation_time = []
    all_future_time = []
    all_coco_time = []
    all_update_time = []
    all_sample_time = []
    all_future_data_time = []
    all_update_data_time = []

    # testing and model saving configs
    test_period = 500
    num_test_runs = 10
    test_render_period = 500
    model_save_period = 500

    episode_count = 0
    frame_count = 0
    episode_reward_history = []

    # create the learning system
    memory = BasicMemory(memory_size=100000)
    p1_learner = RvBLearner(num_joint_actions=number_joint_actions, viewport_size=9)
    p2_learner = RvBLearner(num_joint_actions=number_joint_actions, viewport_size=9)

    # uncomment these lines if you want to restart training from a certain checkpoint
    # run_number = 215000
    # epsilon = 0.11531807651263774
    # episode_count = run_number
    # p1_learner.load_model(f'./models/{run_number}/ddg_model')
    # p2_learner.load_model(f'./models/{run_number}/sag_model')

    config_file = './configs/experiment_2.3_config.ini'
    simulator = RvBSimulation(memory, p1_learner, p2_learner, number_individual_actions, epsilon=epsilon, epsilon_decay=epsilon_decay, config_file=config_file)

    terminal_state_value = np.zeros(number_joint_actions)
    zeros = np.zeros(number_joint_actions)

    # logging setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './logs/' + current_time
    tb_logger = tf.summary.create_file_writer(log_dir)

    # logging metrics
    epsilon_metric = tf.keras.metrics.Mean('epsilon', dtype=tf.float32)
    test_reward_metric = tf.keras.metrics.Mean('test_reward', dtype=tf.float32)
    running_reward_metric = tf.keras.metrics.Mean('running_reward', dtype=tf.float32)
    ddg_loss_metric = tf.keras.metrics.Mean('loss_ddg', dtype=tf.float32)
    sag_loss_metric = tf.keras.metrics.Mean('loss_sag', dtype=tf.float32)
    ddg_avg_error_metric = tf.keras.metrics.Mean('avg_error_ddg', dtype=tf.float32)
    sag_avg_error_metric = tf.keras.metrics.Mean('avg_error_sag', dtype=tf.float32)
    ddg_max_error_metric = tf.keras.metrics.Mean('max_error_ddg', dtype=tf.float32)
    sag_max_error_metric = tf.keras.metrics.Mean('max_error_sag', dtype=tf.float32)
    ddg_gradient_norm_metric = tf.keras.metrics.Mean('max_gradient_norm_ddg', dtype=tf.float32)
    sag_gradient_norm_metric = tf.keras.metrics.Mean('max_gradient_norm_sag', dtype=tf.float32)

    # cache and house keeping
    coco_cache = {}
    future_cache = {}
    all_times = []

    # we need to prime the data pump by running enough games to populate the memory
    print('priming the data pump...')
    while memory.num_samples() < batch_size:
        simulator.run_game_iteration(use_epsilon_greedy=True, render_video=False, video_name='video.mp4', learn=False)
    print('done')

    print('starting the training process')
    with tb_logger.as_default():
        while True:
            start_main = time.time()
            start_simulation = time.time()
            # get one run of the simulator
            episode_reward, number_of_steps = simulator.run_game_iteration(use_epsilon_greedy=True, use_a_star=episode_count % 4 == 0, learn=False)
            end_simulation = time.time()

            episode_count += 1
            frame_count += number_of_steps

            # start the learning event
            if memory.num_samples() > batch_size:

                # 1. pull from the replay buffers all the information needed to do a learning step
                sample_start = time.time()
                states, p1_actions, p2_actions, p1_rewards, p2_rewards, state_primes, dones = memory.sample_memory(count=batch_size)
                sample_end = time.time()

                # 2. predict what the payoff matrices are
                # Note: it is important to note this is with the target model, this is what stabilizes the entire thing
                future_data_start = time.time()
                ddg_dim_observation, sag_dim_observation, combined_goal_heading_vector = get_queryable_state_info(state_primes)
                future_data_end = time.time()

                future_start = time.time()
                p1_future_rewards = p1_learner.predict_with_target_model([ddg_dim_observation, combined_goal_heading_vector])
                p2_future_rewards = p2_learner.predict_with_target_model([sag_dim_observation, combined_goal_heading_vector])
                future_end = time.time()

                # 3. correct all of the terminal states to be all
                # we aren't actually learning terminal state values because no actions are performed from them
                # but COCO-Q has to calculate values from them
                p1_tmp_rewards = np.array(p1_rewards)
                p2_tmp_rewards = np.array(p2_rewards)

                # we are defining the need for the "zeros" when the agent has reached the goal
                # this will make it so there is no advantage and the other player is able to get full reward
                p1_future_rewards[p1_tmp_rewards >= 0] = zeros
                p2_future_rewards[p2_tmp_rewards >= 0] = zeros

                # 4. calculate the coco values for each player
                coco_start = time.time()
                try:
                    p1_coco_values, p2_coco_values = get_stacked_coco_values(p1_future_rewards, p2_future_rewards,
                                                                             coco_cache=coco_cache,
                                                                             shape=(number_individual_actions, number_individual_actions))
                except ValueError as e:
                    error_index = e.args[0]
                    print('*************************')
                    print('ddg_dim_observation')
                    print(ddg_dim_observation[error_index])
                    print('sag_dim_observation')
                    print(sag_dim_observation[error_index])
                    print('combined_goal_heading_vector')
                    print(combined_goal_heading_vector[error_index])

                    raise e


                coco_end = time.time()
                # print(f'coco took {end - start}')

                # 5. calculate the Q-values to be learned
                # Note: this is not the normal Q update formula because this is a neural network, it is directly
                # learning the values rather than using previous values to slightly update the known values
                p1_updated_q_values = p1_rewards + gamma * p1_coco_values
                p2_updated_q_values = p2_rewards + gamma * p2_coco_values

                # 6. calculate the index of the actual joint action that was played
                # Note: the model is outputting a 9x1 vector but everything works off of a 3x3 matrix
                # this converts the index from the 3x3 to the 9x1
                # Note: instead of being 9x1, outputs are (num ddg actions * num sag actions)x1
                # Note: instead of being 3x3, it is (num ddg actions, num sag actions)
                taken_joint_actions = [p1 * number_individual_actions + p2 for p1, p2 in zip(p1_actions, p2_actions)]

                # 7. create the masks that will be used to make sure we only learn from the actions we took
                p1_masks = tf.one_hot(taken_joint_actions, number_joint_actions)
                p2_masks = tf.one_hot(taken_joint_actions, number_joint_actions)

                # 8. actually perform the update
                update_data_start = time.time()
                state_ddg_dim_observation, state_sag_dim_observation, state_combined_goal_heading_vector = get_queryable_state_info(states)
                update_data_end = time.time()

                update_start = time.time()
                p1_learner.update([state_ddg_dim_observation, state_combined_goal_heading_vector], p1_masks, p1_updated_q_values, learner_name='ddg', epoch=episode_count)
                p2_learner.update([state_sag_dim_observation, state_combined_goal_heading_vector], p2_masks, p2_updated_q_values, learner_name='sag', epoch=episode_count)
                update_end = time.time()

                main_end = time.time()

                simulator.decay_epsilon()

                # Update running reward to check condition for solving
                # TODO: is there a way to automatically complete training and quit?
                if episode_count % test_period == 0:

                    # periodically we need to save off the models
                    if episode_count % model_save_period == 0:
                        p1_save_file = f'./models/{episode_count}/ddg_model'
                        p2_save_file = f'./models/{episode_count}/sag_model'

                        p1_learner.save_model(p1_save_file)
                        p2_learner.save_model(p2_save_file)

                    tf.summary.scalar('epsilon', simulator.epsilon, step=episode_count)
                    # network_evaluator.evaluate(f'./models/{episode_count}')
                    # network_evaluator.finished()

                    time_test_start = time.time()

                    # TODO: probably need to decay much more frequently as it never really stops full exploration
                    # simulator.decay_epsilon()

                    total_reward = 0
                    test_run_values = []

                    # run a number of test runs and get the average score to report how we are doing
                    for i in range(num_test_runs):
                        render_video = episode_count % test_render_period == 0 and i == 0
                        test_reward, test_number_of_steps = simulator.run_game_iteration(use_epsilon_greedy=False,
                                                                                         render_video=render_video,
                                                                                         video_name='./videos/tmp_video.mp4',
                                                                                         learn=False)
                        total_reward += test_reward
                        test_run_values.append(test_reward)
                        if render_video:
                            file_name = f'./videos/{episode_count}_{test_reward}_video.mp4'
                            if os.path.exists(file_name):
                                os.remove(file_name)
                            os.rename('./videos/tmp_video.mp4', file_name)

                    # manage the rolling reward history
                    if len(episode_reward_history) > 20:
                        del episode_reward_history[:1]

                    episode_reward_history.append(total_reward / num_test_runs)
                    running_reward = np.mean(episode_reward_history)

                    # tensorboard logging
                    tf.summary.scalar('test_reward', total_reward / num_test_runs, step=episode_count)
                    tf.summary.scalar('running_reward', running_reward, step=episode_count)


                    time_test_end = time.time()
                    time_test_total = time_test_end - time_test_start
                    # print(f'test time {time_test_total}')
                    print(f'test avg reward {(total_reward / num_test_runs):4.2f} running reward {running_reward:.2f} at episode {episode_count}, frame count {frame_count}, epsilon {simulator.epsilon}')
                    print(f'test run results {test_run_values}')
                    print(f'episode reward history {episode_reward_history}')





                # every so often, we need to update the target network to continue learning
                if episode_count % update_target_network == 0:
                    print('swapping brains in target network')
                    p1_learner.update_target_network()
                    p2_learner.update_target_network()

                    coco_cache = {}
                    future_cache = {}

                # some basic time logging
                main_time = main_end - start_main
                simulation_time = end_simulation - start_simulation
                future_time = future_end - future_start
                coco_time = coco_end - coco_start
                update_time = update_end - update_start

                sample_time = sample_end - sample_start
                future_data_time = future_data_end - future_data_start
                update_data_time = update_data_end - update_data_start

                rest = main_time - simulation_time - future_time - coco_time - update_time - sample_time - future_data_time - update_data_time

                all_main_time.append(main_time)
                all_simulation_time.append(simulation_time)
                all_future_time.append(future_time)
                all_coco_time.append(coco_time)
                all_update_time.append(update_time)
                all_sample_time.append(sample_time)
                all_future_data_time.append(future_data_time)
                all_update_data_time.append(update_data_time)

                all_times.append(main_time)
                mean_time = sum(all_times) / len(all_times)

                if len(all_times) > 1000:
                    all_times.pop(0)

                if episode_count % 50 == 0:
                    print(f'{episode_count} main: {main_time:.3f} simulation: {simulation_time:.3f} future: {future_time:.3f} coco: {coco_time:.3f} update: {update_time:.3f} sample: {sample_time:.3f} future data: {future_data_time:.3f} update data: {update_data_time:.3f} rest: {rest:.3f} mean: {mean_time:.5f}')
                    print(f'averages({len(all_main_time)}) main: {sum(all_main_time) / len(all_main_time)} sim: {sum(all_simulation_time) / len(all_simulation_time)}');
                    print(f'future: {sum(all_future_time) / len(all_future_time)} coco: {sum(all_coco_time) / len(all_coco_time)} update: {sum(all_update_time) / len(all_update_time)}')
                    print(f'sample: {sum(all_sample_time) / len(all_sample_time)} future: {sum(all_future_data_time) / len(all_future_data_time)} update data: {sum(all_update_data_time) / len(all_update_data_time)}')


if __name__ == '__main__':
    run()