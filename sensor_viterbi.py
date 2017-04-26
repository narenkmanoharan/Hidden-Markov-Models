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
    states = ('s1', 's2', 's3')

    pos_obs = ('1', '2', '3')

    obs = (
        '1', '1', '1', '3', '1', '2', '3', '1', '1', '3', '1', '1', '1', '1', '1', '1', '1', '1', '1', '3',
        '2', '2', '1', '3', '2', '2', '1', '3', '2', '2')

    start_prob = np.matrix('1 0 0')

    tran_prob = np.matrix([[0.94117647, 0.02941176, 0.02941176],
                           [0.2, 0.8, 0],
                           [0.2, 0, 0.8]])

    obs_prob = np.matrix([[0.375, 0.375, 0.25],
                          [1, 0, 0.0],
                          [0, 1, 0.0]])

    obs_map = observation_mapping(pos_obs)
    state_map = state_mapping(states)

    path = viterbi(states, obs, start_prob, tran_prob, obs_prob, obs_map, state_map)

    print
    print "Viterbi path for Sensor problem"
    print "=============================================="
    print
    print path
    print
    for x in range(0, len(path) - 1):
        if path[x] == 's1' and path[x + 1] == 's2':
            print "The system went into frozen state at " + str(x+1)
            print
        elif path[x] == 's2' and path[x + 1] == 's1':
            print "The system came back online at " + str(x+1)


if __name__ == '__main__':
    main()
