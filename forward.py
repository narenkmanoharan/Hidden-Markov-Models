import numpy as np


def observation_mapping(obs):
    observation_mapping = {}
    for i, o in enumerate(obs):
        observation_mapping[o] = i
    return observation_mapping


def forward_algo(obs, start_prob, tran_prob, obs_prob, obs_map):

    # Store total number of observations total_stages = len(observations)
    total_stages = len(obs)

    # Initialize Alpha
    ob_ind = obs_map[obs[0]]

    alpha = np.multiply(np.transpose(obs_prob[:, ob_ind]), start_prob)

    # Iteratively find alpha(using knowledge of alpha in the previous stage)
    for curr_t in range(1, total_stages):
        ob_ind = obs_map[obs[curr_t]]
        alpha = np.dot(alpha, tran_prob)
        alpha = np.multiply(alpha, np.transpose(obs_prob[:, ob_ind]))

    # Sum the alpha's over the last stage
    total_alpha = alpha.sum()
    return total_alpha


def main():
    states = ('s1', 's2')
    pos_obs = ('1', '2')
    obs = ('1', '2', '2', '1', '1', '1')
    start_prob_1 = np.matrix([[0.3, 0.7]])
    start_prob_2 = np.matrix([[0.7, 0.3]])
    tran_prob_1 = np.matrix([[0.3, 0.3], [0.7, 0.7]])
    tran_prob_2 = np.matrix([[0.7, 0.7], [0.3, 0.3]])
    obs_prob = np.matrix([[1, 0], [0, 1]])
    obs_map = observation_mapping(pos_obs)
    print
    x = forward_algo(obs, start_prob_1, tran_prob_1, obs_prob, obs_map)
    print "Model 1: " + str(x)
    print
    y = forward_algo(obs, start_prob_2, tran_prob_2, obs_prob, obs_map)
    print "Model 2: " + str(y)



if __name__ == '__main__':
    main()