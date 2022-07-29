import random

import numpy as np
import cv2
import imageio

from RvBEnvironment import RvBEnvironment
from utils import get_actions_from_rvb_state
from RvBUtils import a_star


class RvBSimulation:

    def __init__(self, memory, p1_learner, p2_learner, num_individual_actions, epsilon=0.99, epsilon_decay=0.99999, min_epsilon=0.1, config_file=None):

        # where we will store our experiences
        self.memory = memory

        # learning parameters
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.num_individual_actions = num_individual_actions

        # the models that are being trained
        self.p1_learner = p1_learner
        self.p2_learner = p2_learner

        # the environment we will interact with
        self.env = RvBEnvironment(stationary_drone=False, config_file=config_file)

        # bookkeeping for the rewards
        self.previous_ddg_safe = False
        self.previous_sag_safe = False

    def get_player_actions(self, state, use_epsilon_greedy=True, verbose=False, learn=True):

        # chose an action for each agent to perform
        if use_epsilon_greedy and random.random() < self.epsilon:
            # get random actions to perform
            p1_action = np.random.choice(self.num_individual_actions)
            p2_action = np.random.choice(self.num_individual_actions)
        else:
            # what do the models say to do?
            p1_action, p2_action = get_actions_from_rvb_state(state, self.p1_learner, self.p2_learner, verbose=verbose, learn=learn)

        return p1_action, p2_action

    def run_game_iteration(self, use_epsilon_greedy=True, visualize=False, render_video=False, video_name=None, verbose=False, use_a_star=False, learn=True):

        if render_video:
            video = imageio.get_writer(video_name, fps=24)

        episode_reward = 0
        number_of_steps = 0

        # reset the environment
        state = self.env.reset()
        done = False

        # which ship is which?
        ddg_num = self.env.ddg_number
        sag_num = self.env.sag_number

        # TODO: really don't like having all the visualization and video stuff here.  refactor opportunity??
        if visualize:
            img = self.env.render(mode='human')
            cv2.imshow("Canvas", img)
            cv2.waitKey(0)

        if render_video:
            video.append_data(self.scale_image(cv2.cvtColor(self.env.render(mode='human'), cv2.COLOR_BGR2RGB)))

        if use_a_star:
            p1_path, p2_path = a_star(self.env.game)

        while not done:

            # get actions to apply to the environment
            if use_a_star and p1_path is not None and p2_path is not None:

                if len(p1_path) > number_of_steps:
                    p1_action = p1_path[number_of_steps]
                else:
                    p1_action = 1

                if len(p2_path) > number_of_steps:
                    p2_action = p2_path[number_of_steps]
                else:
                    p2_action = 1

                joint_action = (p1_action, p2_action)

            else:
                joint_action = self.get_player_actions(state, use_epsilon_greedy=use_epsilon_greedy, verbose=verbose, learn=learn)

            p1_action, p2_action = joint_action

            number_of_steps += 1

            # Apply the sampled action to our environment
            state_next, reward, done, _ = self.env.step(joint_action)
            p1_reward, p2_reward = reward

            # add up how much reward we received
            episode_reward += p1_reward + p2_reward

            # experience tuple management
            # TODO: need to make the states a stacked history
            self.memory.remember(state, p1_action, p2_action, p1_reward, p2_reward, state_next, done)

            # the next state is now our current state
            state = state_next

            if visualize:
                img = self.env.render(mode='human')
                cv2.imshow("Canvas", img)
                cv2.waitKey(0)

            if render_video:
                video.append_data(self.scale_image(cv2.cvtColor(self.env.render(mode='human'), cv2.COLOR_BGR2RGB)))

        if render_video:
            video.close()

        # did the ddg and sag make it?
        ddg_safe = False
        sag_safe = False
        for ship in self.env.game.ships_at_dest:
            if ship.number == ddg_num:
                ddg_safe = True
            elif ship.number == sag_num:
                sag_safe = True

        self.previous_ddg_safe = ddg_safe
        self.previous_sag_safe = sag_safe

        return episode_reward, number_of_steps

    def decay_epsilon(self):
        # periodically we need to make the exploration rate lower until we get to min_epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    @staticmethod
    def scale_image(image, scale=512):
        w, h = image.shape[:2]

        x_new = scale
        y_new = scale

        x_scale = x_new / (w - 1)
        y_scale = y_new / (h - 1)

        new_image = np.zeros([x_new, y_new, 3], dtype=np.uint8)

        for i in range(x_new - 1):
            for j in range(y_new - 1):
                new_image[i + 1, j + 1] = image[1 + int(i / x_scale), 1 + int(j / y_scale)]

        return new_image

