import numpy as np


def state_mapping(states):
    state_mapping = {}
    for i, o in enumerate(states):
        state_mapping[i] = o
    return state_mapping


def observation_mapping(obs):
    observation_mapping = {}
    for i, o in enumerate(obs):
        observation_mapping[o] = i
    return observation_mapping


def viterbi(states, obs, start_prob, tran_prob, obs_prob, obs_map, state_map):
    # Find total states,observations
    total_stages = len(obs)
    num_states = len(states)

    # initialize data
    # Path stores the state sequence giving maximum probability
    old_path = np.zeros((total_stages, num_states))
    new_path = np.zeros((total_stages, num_states))

    # Find initial delta
    # Map observation to an index
    # delta[s] stores the probability of most probable path ending in state 's'
    ob_ind = obs_map[obs[0]]
    delta = np.multiply(np.transpose(obs_prob[:, ob_ind]), start_prob)

    # Scale delta
    delta = delta / np.sum(delta)

    # initialize path
    old_path[0, :] = [i for i in range(num_states)]

    # Find delta[t][x] for each state 'x' at the iteration 't'
    # delta[t][x] can be found using delta[t-1][x] and taking the maximum possible path
    for curr_t in range(1, total_stages):

        # Map observation to an index
        ob_ind = obs_map[obs[curr_t]]
        # Find temp and take max along each row to get delta
        temp = np.multiply(np.multiply(delta, tran_prob.transpose()), obs_prob[:, ob_ind])

        # Update delta and scale it
        delta = temp.max(axis=1).transpose()
        delta = delta / np.sum(delta)

        # Find state which is most probable using argax
        # Convert to a list for easier processing
        max_temp = temp.argmax(axis=1).transpose()
        max_temp = np.ravel(max_temp).tolist()

        # Update path
        for s in range(num_states):
            new_path[:curr_t, s] = old_path[0:curr_t, max_temp[s]]

        new_path[curr_t, :] = [i for i in range(num_states)]
        old_path = new_path.copy()

    # Find the state in last stage, giving maximum probability
    final_max = np.argmax(np.ravel(delta))
    best_path = old_path[:, final_max].tolist()
    path = [state_map[i] for i in best_path]

    return path


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

    obs_prob = np.matrix([[0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                          [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                          [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                          [0.167, 0.167, 0.167, 0.167, 0.167, 0.167],
                          [0.05, 0.125, 0.125, 0.125, 0.125, 0.45],
                          [0.05, 0.125, 0.125, 0.125, 0.125, 0.45],
                          [0.05, 0.125, 0.125, 0.125, 0.125, 0.45],
                          [0.05, 0.125, 0.125, 0.125, 0.125, 0.45]])

    obs_map = observation_mapping(pos_obs)
    state_map = state_mapping(states)

    path = viterbi(states, obs, start_prob, tran_prob, obs_prob, obs_map, state_map)

    print
    print "Viterbi path for Casino problem"
    print "=============================================="
    print
    print path
    print
    for x in range(0, len(path) - 1):
        if path[x] == 's4' and path[x + 1] == 's5':
            print "Switching to loaded dice at " + str(x+1)
        elif path[x] == 's8' and path[x + 1] == 's1':
            print "Switching to normal dice at " + str(x+1)


if __name__ == '__main__':
    main()
