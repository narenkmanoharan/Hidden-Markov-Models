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
    states = ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8')

    pos_obs = ('1', '2', '3', '4', '5', '6',)

    obs = (
        '1', '6', '3', '4', '1', '6', '3', '4', '3', '5', '1', '4', '6', '2', '5', '4', '2', '6', '3', '1',
        '5', '5', '2', '6', '4', '4', '2', '5', '1', '6', '2', '3', '4', '1', '3', '1', '6', '3', '1', '5',
        '5', '6', '4', '1', '6', '3', '6', '2', '6', '5', '1', '4', '2', '5', '6', '3', '4', '6', '2', '6',
        '3', '5', '6', '2', '6', '5', '3', '6', '1', '4', '5', '2', '6', '3', '5', '2', '6', '1', '5', '3',
        '4', '1', '6', '2', '6', '3', '5', '2')

    start_prob = np.matrix('1 0 0 0 0 0 0 0')

    tran_prob = np.matrix([[0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0.75, 0, 0, 0, 0.25, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0.25, 0, 0, 0, 0.75, 0, 0, 0]])

    obs_prob_cheating = np.matrix([[0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                                   [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                                   [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                                   [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                                   [0.05, 0.125, 0.125, 0.125, 0.125, 0.45],
                                   [0.05, 0.125, 0.125, 0.125, 0.125, 0.45],
                                   [0.05, 0.125, 0.125, 0.125, 0.125, 0.45],
                                   [0.05, 0.125, 0.125, 0.125, 0.125, 0.45]])

    obs_prob_fair = np.matrix([[0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                               [0.167, 0.167, 0.167, 0.167, 0.167, 0.167]])

    obs_map = observation_mapping(pos_obs)
    x = forward_algo(obs, start_prob, tran_prob, obs_prob_cheating, obs_map)
    y = forward_algo(obs, start_prob, tran_prob, obs_prob_fair, obs_map)

    print
    print "Forward Algorithm for casino problem"
    print "=============================================="
    print
    print "Regular Casino: " + str(y)
    print "Cheating Casino: " + str(x)
    print
    print "Regular casino is more probable to produce the sequence" if y > x else "Cheating casino is more probable to produce the sequence"
    print


if __name__ == '__main__':
    main()
