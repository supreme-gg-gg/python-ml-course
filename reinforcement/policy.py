# Policy iteration

import numpy as np

# These can also be used as function arguments for reusability
# NUM_STATES and NUM_ACTION represent the state and action space respectively
NUM_STATES = 5
NUM_ACTIONS = 2
GAMMA = 0.9

def policy_iteration(rewards, transition_prob):

    '''
    Evaluate policy: update the V values of the policy at each state
    Improve policy: update the policy by finding the a that max Q(s,a)
    Repeat to update V values again of the new policy in next evaluate
    Reapte until convergence (policy IS stable)
    '''

    # Initialize policy randomly
    policy = np.zeros(NUM_STATES, dtype=int)

    max_policy_iter = 10_000

    for _ in range(max_policy_iter):
        policy_stable = True

        # Policy evaluation
        # V is the value function of the current policy
        V = evaluate_policy(rewards, transition_prob, policy)

        # Policy improvement
        policy, policy_stable = improve_policy(rewards, transition_prob, policy, V)

        if policy_stable:
            break
    
    return policy, V

def evaluate_policy(rewards, transition_prob, policy):

    '''
    Evaluate the value function of the current policy, 
    returns a list of values for each state
    '''

    max_policy_eval = 10_000
    threshold = 1e-10

    V = np.zeros(NUM_STATES)

    for _ in range(max_policy_eval):
        delta = 0
        for s in range(NUM_STATES):
            v = V[s]
            V[s] = calculate_Q_value(rewards, transition_prob, V, s, policy[s])
            delta = max(delta, abs(v - V[s]))

        if delta < threshold: 
            break

    return V

def improve_policy(rewards, transition_prob, policy, V):

    '''
    Improve the policy by selecting the action that maximizes the Q value
    '''

    policy_stable = True

    for s in range(NUM_STATES):
        q_best = V[s]

        for a in range(NUM_ACTIONS):
            q_sa = calculate_Q_value(rewards, transition_prob, V, s, a)

            if q_sa > q_best and policy[s] != a:
                q_best = q_sa
                policy[s] = a
                policy_stable = False

    return policy, policy_stable

def calculate_Q_value(rewards, transition_prob, V, s, a):
    
    '''
    Calculates the Q value for a given state and action
    '''

    q = 0
    q += rewards[s]
    for s_prime in range(NUM_STATES):
        q += GAMMA * transition_prob[s, a, s_prime] * V[s_prime]

    return q

def generate_rewards(each_step_reward, terminal_left_reward, terminal_right_reward):
    
    # This can be constructed in a way that suits your problem
    
    rewards = np.array([each_step_reward] * NUM_STATES)
    rewards[0] = terminal_left_reward
    rewards[-1] = terminal_right_reward

    return rewards

def generate_transition_prob(misstep_prob=0):
    
    # There are only two possible actions: 0 = left, 1 = right
    
    ''' 
    Output: p[s, a, s'] = P(s' | s, a)
    transition_prob is a 3D array where the first dimension is the current state,
    the second dimension is the action, and the third dimension is the next state

    For example, transition_prob[0, 0, 0] = 0.8 means that the probability of moving
    from state 0 to state 0 by taking action 0 (left) is 0.8
    '''
    
    p = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

    for i in range(NUM_STATES):

        # if it is not the terminal state left or right
        # since the terminal states have no transitions (i-1 or i+1)

        # e.g. if i = 1, then p[1, 0, 0] = 0.8 and p[1, 0, 2] = 0.2
        # and p[1, 0, 1], p[1, 0, 3], p[1, 0, 4] are all zero

        if i != 0:
            p[i, 0, i-1] = 1 - misstep_prob
            p[i, 1, i-1] = misstep_prob
        
        if i != NUM_STATES - 1:
            p[i, 1, i+1] = 1 - misstep_prob
            p[i, 0, i+1] = misstep_prob

    # Terminal states
    p[0] = np.zeros((NUM_ACTIONS, NUM_STATES))
    p[-1] = np.zeros((NUM_ACTIONS, NUM_STATES))

    return p

# This is only run if the script is run directly, not when it's imported
if __name__ == '__main__':

    rewards = np.array([1,0,0,0,10])

    # You should know how to interpret this, see the function above
    transition_prob = np.array([
        [[0.8, 0.2, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0, 0.0],
         [0.8, 0.1, 0.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.8, 0.2, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.8, 0.2],
         [0.0, 0.0, 0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 1.0]]
    ])

    policy, V = policy_iteration(rewards, transition_prob)
    print(policy, V)

