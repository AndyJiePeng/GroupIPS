"""
    Implementation of InversePropensityWeighting method

"""
import numpy as np


class IPW:
    def __init__(self, context, treatment, reward):
        self.context = context
        self.treatment = treatment
        self.reward = reward

    def estimate_reward(self, reward, action, pscore, target_policy, policy_type="uniform"):
        size = self.context.shape[0]
        action = action.astype('int32')

        # Calculating the inverse_propensity and the rewards
        target_weight = target_policy[np.arange(size), np.squeeze(action)]
        propensity = pscore[np.arange(size), np.squeeze(action)]
        inverse_propensity = target_weight / propensity
        print("Variance of IPW:", np.var(inverse_propensity))
        # print(inverse_propensity)
        # Clipping
        lambda_min = 0.05
        lambda_max = 0.95
        # if isinstance(inverse_propensity, np.ndarray):
        #     inverse_propensity = np.maximum(inverse_propensity, lambda_min)
        #     inverse_propensity = np.minimum(inverse_propensity, lambda_max)

        return (reward.T * inverse_propensity)
