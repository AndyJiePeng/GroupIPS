"""
    Implementation of Self-Normalized InversePropensityWeighting method

"""
import numpy as np


class SNIPW:
    def __init__(self, context, treatment, reward):
        self.context = context
        self.treatment = treatment
        self.reward = reward

    def estimate_reward(self, reward, action, pscore, target_policy, policy_type="uniform"):
        size = self.context.shape[0]
        action = action.astype('int32')

        # Calculating the inverse_propensity and the rewards
        # Target
        target_weight = target_policy[np.arange(size), np.squeeze(action)]
        propensity = pscore[np.arange(size), np.squeeze(action)]

        inverse_propensity = target_weight / propensity
        # print(inverse_propensity)
        return (reward.T * inverse_propensity / inverse_propensity.mean())
