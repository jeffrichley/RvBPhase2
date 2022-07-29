from RvBSimulation import RvBSimulation
from Memory import BasicMemory
from RvBLearner import RvBLearner

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

number_individual_actions = 4
number_joint_actions = number_individual_actions * number_individual_actions
epsilon = 0.99

memory = BasicMemory(memory_size=1000000)
p1_learner = RvBLearner(num_joint_actions=number_joint_actions, viewport_size=9)
p2_learner = RvBLearner(num_joint_actions=number_joint_actions, viewport_size=9)
config_file = './configs/test100-config.ini'
simulator = RvBSimulation(memory, p1_learner, p2_learner, number_individual_actions, epsilon=epsilon, epsilon_decay=0.99999, config_file=config_file)

run_number = 99000

p1_learner.load_model(f'./models/{run_number}/ddg_model')
p2_learner.load_model(f'./models/{run_number}/sag_model')

run_many = True

if run_many:
    all_rewards = []
    for i in range(2000):
        test_reward, test_number_of_steps = \
            simulator.run_game_iteration(use_epsilon_greedy=False, visualize=False,
                                         render_video=True, video_name='./test_vids/tmp_video.mp4',
                                         verbose=False, use_a_star=False, learn=False)
        if os.path.exists('./test_vids/tmp_video.mp4'):
            os.rename('./test_vids/tmp_video.mp4', f'./test_vids/{i}-{test_reward}.mp4')
        all_rewards.append(test_reward)
        print(test_reward, test_number_of_steps)

    print(f'average reward {sum(all_rewards) / len(all_rewards)}')

else:
    test_reward, test_number_of_steps = \
        simulator.run_game_iteration(use_epsilon_greedy=False, visualize=True,
                                        render_video=False, video_name='./videos/tmp_video.mp4',
                                        verbose=True, use_a_star=False)
    print(test_reward, test_number_of_steps)

