import numpy as np
import networkx as nx
#import tensorflow as tf
import gym

# AoI environment definition
class aoiEnv(gym.Env):      # uncomment this for openai gym compatibility, and comment out next line
#class aoiEnv():

    def __init__(self, A, reward_indicator, iaa_limit, ipa_limit, max_steps_per_episode = None):
        
        #  Inputs --
        #                     A : a square adjancency matrix describing the network topology
        #      reward_indicator : a matrix of same size as A, all elements 0 or 1.  If (i,j) entry is 1, 
        #                           the age of node i's knowledge of process j is counted
        #                           in the reward.  The diagonal entries are all assumed to be zero since
        #                           a node always has "ageless" knowledge of its own process.
        #             iaa_limit : scalar, when inst avg age exceeds this quantity, episode is done. This
        #                           parameter needs to be chosen somewhat carefully, and is dependent on
        #                           the prior two parameters
        #             ipa_limit : scalar, when inst peak age exceeds this quantity, episode is done. 
        # max_steps_per_episode : optional, ignored if "None" or not provided.  Can be used to limit episode length.
        
        # determine observation space (and mapping) based on A and reward_indicator
        N = A.shape[0]  # size of A
        status_no_reward = np.sum(reward_indicator, axis=0)==0  # determine if there are any statuses that we don't care about (not part of reward)
        tmpObs = np.full((N,N), 1) - np.identity(N)             # start with everything, except each node's own status (age=0)
        tmpObs[:,status_no_reward] = 0                          # remove statuses that we don't care about 
        tmpObs = np.unravel_index(np.nonzero(tmpObs.flatten()), (N,N))  # convert to indices
        self.observation_idx = np.concatenate((tmpObs[0],tmpObs[1]),0).transpose()  # save observation space tuples here
        self.num_observations = self.observation_idx.shape[0]                       # number of observations (statuses)
        self.observation_space = gym.spaces.Box(low=1, high=ipa_limit, shape=(self.num_observations, ), dtype=np.uint32)   # uncomment this for openai gym compatibility
        #self.observation_space = gym.spaces.MultiDiscrete([ipa_limit+1] * self.num_observations)   # uncomment this for openai gym compatibility

        # determine action space (and mapping) based on A and reward_indicator
        tmpAct = np.full((N,N), 1)                              # start with everything
        tmpAct[np.sum(A, axis=0)==1,:]=0                        # remove true leaf nodes
        np.fill_diagonal(tmpAct, 1)                             # but allow leaf nodes to disseminate their own status
        tmpAct[:,status_no_reward] = 0                          # remove statuses that we don't care about 
        tmpAct = np.unravel_index(np.nonzero(tmpAct.flatten()), (N,N))  # convert to indices
        self.action_idx = np.concatenate((tmpAct[0],tmpAct[1]),0).transpose()       # save action space tuples here
        self.num_actions = self.action_idx.shape[0]                 # number of actions
        self.action_space = gym.spaces.Discrete(self.num_actions)  # uncomment this for openai gym compatibility
        
        # set up static variables we need to preserve
        self.A = A
        self.reward_idx = reward_indicator[tmpObs][0].astype(bool)  # indicates which elements of state vector are part of reward
        self.iaa_limit = iaa_limit                                  # largest allowable inst. average age
        self.ipa_limit = ipa_limit                                  # largest allowable inst. peak age (optional, unused if set to None)
        self.max_steps_per_episode = max_steps_per_episode          # episode length limit (can be None)
        
        # compute shortest path lengths for all nodes, use as optimistic initialization
        self.state_init = np.zeros((self.num_observations, ), dtype=np.uint32)       # allocate space
        # can comment next three lines to use all zeros as initialization.  This can be nice since running reward is then equal to negative inst avg age.
        p = dict(nx.shortest_path_length(nx.from_numpy_matrix(A)))  # get shortest path generator
        for i in range(self.num_observations):                      # extract relevant paths from generator
            self.state_init[i] = p[self.observation_idx[i,0]][self.observation_idx[i,1]]

        # initialize internal states
        self.action = None
        self.reward = None
        self.steps = 0
        self.state = self.state_init                         # initialize state vector
        self.iaa = np.mean(self.state[self.reward_idx])      # initialize instantaneous average age
        self.ipa = np.max(self.state[self.reward_idx])       # initialize instantaneous peak age
        return None

    def step(self, action):  # this function implements equation (1) in our JCN paper
        self.state = self.state + 1                  # all statuses get older by 1
        self.steps += 1                              # increment step counter
        self.action = action
        i = self.action_idx[action,0]                # current action concerns node i's estimate of process j...
        j = self.action_idx[action,1]                #    ... so we extract i and j
        cur_idx = np.where((self.observation_idx == [i,j]).all(axis=1))[0]  # status table index that concerns current action (doesn't exist if i=j)
        for obs in np.nditer(np.nonzero(self.observation_idx[:,1] == j)[0]):  # loop through status table entries concerning process j
            m = self.observation_idx[obs,0]          # this entry in status table concerns node m
            if self.A[m,i]:                          # if m and i are neighbors, and ... 
                if i == j:                                  # ... i disseminated its own process ==> age is 1
                    self.state[obs] = 1
                elif self.state[cur_idx] < self.state[obs]: # ... or, the dissmeminated process is "fresh" ==> update
                    self.state[obs] = self.state[cur_idx]

        temp = self.iaa                                  # save old age
        self.iaa = np.mean(self.state[self.reward_idx])  # calculate new inst avg age
        self.ipa = np.max(self.state[self.reward_idx])   # calculate new inst peak age
        self.reward = temp - self.iaa                         # compute reward (negative change in IAA)
        done = (self.iaa > self.iaa_limit) or (self.ipa > self.ipa_limit)    # if inst avg age exceeds limit, we're done
        if self.max_steps_per_episode is not None and self.steps >= self.max_steps_per_episode:   # if an episode length limit was specified, check and terminate if exceeded
            done = True
        #if self.reward == -1:
        #    done = True                                  # episode ends if no statuses updated (bad move!)
        info = {}                                         # unused, required by openai gym
        return self.state, self.reward, bool(done), info

    def reset(self):
        # re-initialize internal states
        self.action = None
        self.reward = None
        self.steps = 0
        self.state = self.state_init                         # initialize state vector
        self.iaa = np.mean(self.state[self.reward_idx])      # initialize instantaneous average age
        self.ipa = np.max(self.state[self.reward_idx])       # initialize instantaneous peak age
        return self.state

    def render(self):
        print('action taken: (', self.action_idx[self.action,0], ',', self.action_idx[self.action,1],')')
        print('reward: {:.1f}'.format(self.reward))
        print('instantaneous avg age: {:.1f}'.format(self.iaa))
        print('instantaneous peak age: {:.1f}'.format(self.ipa))
        print('')
        print('(node, process)   age')
        print('---------------------')
        for i in range(self.num_observations):
            print('(',self.observation_idx[i,0],',',self.observation_idx[i,1],')','        ', self.state[i])
        print('')
        return None

def print_results(env, model, steps_to_display, steps_to_include_in_age_calc, supress_output = False):
    iaa_history = []
    ipa_history = []
    obs = env.reset()
    done = False
    step = 0
    terminate = max([steps_to_display]+steps_to_include_in_age_calc)
    action=np.zeros(terminate+1, dtype=np.uint32)
    if not supress_output:
        print('')
        print('Example output from beginning of an episode (with idealized initial condition) --')
        print('')
    while not done and step < terminate:
        step += 1
        # action_probs = model(tf.convert_to_tensor(np.expand_dims(obs, axis=0)), training=False)
        # action[step] = tf.argmax(action_probs[0]).numpy()
        action[step], _ = model.predict(obs)
        obs, reward, done, info = env.step(action[step])
        if step in steps_to_include_in_age_calc:
            iaa_history.append(env.iaa)
            ipa_history.append(env.ipa)
        if step <= steps_to_display and not supress_output:
            env.render()
    if step >= max(steps_to_include_in_age_calc):
        avg_age = np.mean(iaa_history)+0.5
        if not supress_output:
            print('Over specified steps, the ages are --')
            print('Average age: {:.3f}'.format(avg_age))
            print('Peak age: {}'.format(np.max(ipa_history)+1))
            for period in range(2, (terminate+1)//2):
                if np.all(action[-period:] == action[-2*period:-period]):  # check for periodicity 
                    print('')
                    print('*********************************************************')
                    print('Schedule is periodic with period', period, 'as follows --')
                    for j in range(period):
                        print('(', env.action_idx[action[-period+j],0], ',', env.action_idx[action[-period+j],1],')')
                    print('... repeat.  Over one period, the ages are --')
                    print('Average age: {:.3f}'.format(np.mean(iaa_history[-period:])+0.5))
                    print('Peak age: {}'.format(np.max(ipa_history[-period:])+1))
                    print('*********************************************************')
                    print('')
                    break
    else:
        avg_age = np.inf  # terminated early
    return None
