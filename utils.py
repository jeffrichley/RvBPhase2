import numpy as np
import itertools
from numpy import unravel_index
from multiprocessing import Pool

from cvxopt import matrix, solvers
from cvxopt.solvers import options

# make CVXOPT quiet
options['show_progress'] = False
solvers.options['show_progress'] = False  # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions


def get_coco_value(player1_vals, player2_vals):

    plus = player1_vals + player2_vals
    max_value = (plus / 2).max()

    minus = player1_vals - player2_vals
    minimax_probability_value = minimax_value(minus/2)

    coco_value = max_value + minimax_probability_value

    return coco_value


def get_coco_values(player1_vals, player2_vals):

    plus = player1_vals + player2_vals
    max_value = (plus / 2).max()

    minus = player1_vals - player2_vals
    minimax_probability_value = minimax_value(minus/2)

    p1_coco_value = max_value + minimax_probability_value
    p2_coco_value = max_value - minimax_probability_value

    return p1_coco_value, p2_coco_value


def maxmin(A, solver="glpk"):

    original_A = A.copy()

    # https://adamnovotnycom.medium.com/linear-programming-in-python-cvxopt-and-game-theory-8626a143d428

    num_vars = len(A)
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)

    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T  # reformat each variable is in a row
    G *= -1  # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1])  # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1)  # insert utility column
    G = matrix(G)
    h = ([0 for i in range(num_vars)] +
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)

    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)

    # TODO: why does it crash sometimes?  I've noticed there are times when all values are negative.
    # This may cause the LP to become unbounded.
    if sol['x'] is None:
        print(original_A)

    probs = sol['x'][1:]

    return probs


def minimax_value(vals):
    mm1 = maxmin(vals)
    mm2 = maxmin(-vals.transpose())
    probs = np.outer(mm1, mm2)
    tots = vals * probs
    answer = tots.sum()

    return answer


def get_actions_from_state(state, p1_learner, p2_learner):

    expanded_state = np.expand_dims(state, axis=0)
    p1_action_values = p1_learner.predict(expanded_state)
    p2_action_values = p2_learner.predict(expanded_state)
    action_values = np.reshape(p1_action_values + p2_action_values, (3, 3))
    p1_action, p2_action = unravel_index(action_values.argmax(), action_values.shape)

    return p1_action, p2_action


def agents_have_shared_state(state):
    p1_observation = state[0]

    # we need to look at p1's observation and check to see if the first layer has a 1 (player 1) and the second layer
    # has a 1 (player 2).  If both have a 1, they are close enough to be sharing their observations
    return min(p1_observation[:, :, 0].max(), p1_observation[:, :, 1].max()) == 1

def get_actions_from_rvb_state(state, p1_learner, p2_learner, verbose=False, learn=True, p1_num_actions=4, p2_num_actions=4):

    # ddg_dim_observation, ddg_goal_vector, ddg_heading, ddg_missile_count, sag_dim_observation, sag_goal_vector, sag_heading, sag_missile_count = state
    ddg_dim_observation, ddg_goal_vector, ddg_heading, sag_dim_observation, sag_goal_vector, sag_heading = state
    expanded_ddg_dim_observation = np.expand_dims(ddg_dim_observation, axis=0)
    expanded_sag_dim_observation = np.expand_dims(sag_dim_observation, axis=0)
    # expanded_unit_vector_heading_combo = np.expand_dims(np.concatenate([ddg_goal_vector, ddg_heading, sag_goal_vector, sag_heading, [ddg_missile_count, sag_missile_count]]), axis=0)
    expanded_unit_vector_heading_combo = np.expand_dims(np.concatenate([ddg_goal_vector, ddg_heading, sag_goal_vector, sag_heading]), axis=0)

    p1_action_values = p1_learner.predict([expanded_ddg_dim_observation, expanded_unit_vector_heading_combo], training=learn)
    p2_action_values = p2_learner.predict([expanded_sag_dim_observation, expanded_unit_vector_heading_combo], training=learn)

    if verbose:
        print('---------------------------------')
        print(np.reshape(p1_action_values, (p1_num_actions, p2_num_actions)))
        print(np.reshape(p2_action_values, (p1_num_actions, p2_num_actions)))
        print(np.reshape(p1_action_values + p2_action_values, (p1_num_actions, p2_num_actions)))
        print('---------------------------------')

    if agents_have_shared_state(state):
        # if the agents are close enough, we can do a normal joint action
        action_values = np.reshape(p1_action_values + p2_action_values, (p1_num_actions, p2_num_actions))
        p1_action, p2_action = unravel_index(action_values.argmax(), action_values.shape)
    else:
        # if they aren't close enough, we will just take each agent's maximizing action
        p1_action_values = np.reshape(p1_action_values, (p1_num_actions, p2_num_actions))
        p2_action_values = np.reshape(p2_action_values, (p1_num_actions, p2_num_actions))

        p1_action = np.argmax(np.max(p1_action_values, axis=1))
        p2_action = np.argmax(np.max(p2_action_values, axis=0))

    return p1_action, p2_action


def get_stacked_coco_values(p1_payoffs, p2_payoffs, shape=(3, 3), coco_cache=None):

    p1_coco_values = np.zeros(len(p1_payoffs))
    p2_coco_values = np.zeros(len(p1_payoffs))

    for i in range(len(p1_payoffs)):
        p1_payoff = np.reshape(p1_payoffs[i], shape)
        p2_payoff = np.reshape(p2_payoffs[i], shape)

        key = None
        found = False

        if coco_cache is not None:
            key = tuple(np.concatenate([p1_payoffs[i], p2_payoffs[i]]).tolist())
            if key in coco_cache:
                p1_coco, p2_coco = coco_cache[key]
                found = True

        if not found:
            try:
                p1_coco, p2_coco = get_coco_values(p1_payoff, p2_payoff)
            except:
                print(f'cocos {i}')
                print('p1_payoff')
                print(p1_payoff)
                print('p2_payoff')
                print(p2_payoff)

                raise ValueError(i)

            if coco_cache is not None:
                coco_cache[key] = (p1_coco, p2_coco)

        p1_coco_values[i] = p1_coco
        p2_coco_values[i] = p2_coco

    return p1_coco_values, p2_coco_values


def partial_stacked_coco(values):

    p1_payoffs, p2_payoffs, shape, coco_cache = values

    p1_payoff = np.reshape(p1_payoffs, shape)
    p2_payoff = np.reshape(p2_payoffs, shape)

    key = None
    found = False

    if coco_cache is not None:
        key = tuple(np.concatenate([p1_payoffs, p2_payoffs]).tolist())
        if key in coco_cache:
            p1_coco, p2_coco = coco_cache[key]
            found = True

    if not found:
        p1_coco, p2_coco = get_coco_values(p1_payoff, p2_payoff)

        if coco_cache is not None:
            coco_cache[key] = (p1_coco, p2_coco)

    return p1_coco, p2_coco


def get_queryable_state_info(state_info):
    ddg_dim_observation = np.array([x[0] for x in state_info])
    ddg_goal_vector = [x[1] for x in state_info]
    ddg_heading_vector = [x[2] for x in state_info]
    # ddg_missile_counts = [x[3] for x in state_info]
    sag_dim_observation = np.array([x[3] for x in state_info])
    sag_goal_vector = [x[4] for x in state_info]
    sag_heading_vector = [x[5] for x in state_info]
    # sag_missile_counts = [x[7] for x in state_info]
    combined_goal_heading_vector = np.array([[w[0], w[1], x[0], x[1], y[0], y[1], z[0], z[1]]
                                             for w, x, y, z in
                                             zip(ddg_goal_vector, ddg_heading_vector,
                                                 sag_goal_vector, sag_heading_vector)])

    return ddg_dim_observation, sag_dim_observation, combined_goal_heading_vector
