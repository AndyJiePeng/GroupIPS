"""
    # Policy Evaluation - complex treatment
    # Author: Jie Peng
    # Date: Jan 2022
"""
import argparse
from syntheticData import Synthetic_data
from trainer import run_latent_method
from utils import validate_results, save_model_file, treatment_effect_estimation
import numpy as np
from sklearn.neural_network import MLPRegressor
import torch
from regression import cal_propensity_score
import warnings
import torch.nn.functional as F
from model import IPW, SNIPW, DirectMethod, DoublyRobust
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering,KMeans 
from scipy import stats
from copy import copy
from tqdm import tqdm
from obp.ope import OffPolicyEvaluation
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import DoublyRobust as DR
from obp.ope import DirectMethod as DM
from obp.ope import SwitchDoublyRobust as Switch
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPS
from obp.ope import RegressionModel
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")
# Global Variable
K = 20  # K as the iteration to calculate Bias/SD/MSE
VERBOSE = False
GROUP_NUM = [2,3,5,10,20,50,80,100]


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_d", default=10, type=int)
    parser.add_argument("--size", default=5000, type=int)
    parser.add_argument("--num_contexts", default=10, type=int)
    parser.add_argument("--behavior_policy", default='inverse', type=str)
    parser.add_argument("--target_policy", default='sigmoid', type=str)
    parser.add_argument("--gpu", default='7', type=str)
    parser.add_argument("--interaction", default=True, type=bool)
    parser.add_argument("--truePropensity", default=True, type=bool)
    parser.add_argument("--model_z", default=1, type=int)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--epislon", default=0.1, type=float)
    parser.add_argument("--group_k", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--treatmentEffect", default=0.8, type=float)
    parser.add_argument("--filePath", default=None, type=str)
    parser.add_argument("--visual", default=0, type=int)
    parser.add_argument("--std_strength", default=1, type=float)
    parser.add_argument("--vary_group", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--embedding_size", default=2, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=800, type=int)

    return parser.parse_args()


def main():
    sigmoid = False
    global args
    args = init_arg()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # args.group_k = int(args.t_d / 5)


    # treatment_effect_estimation()

    # Generate the data based on the behavior policy and rewards
    Simulated_data = Synthetic_data(size=args.size,
                                    x_dimension=args.num_contexts,
                                    t_d=args.t_d,
                                    seed=args.seed,
                                    std_strength=args.std_strength,
                                    vary_group=args.vary_group)

    # Generate the behavior policy based on the value of beta
    behavior_policy = Simulated_data.generate_behavior_policy(beta=args.beta, 
                                                            treatmentEffect=args.treatmentEffect)

    # Generate the treatment based on the behavior policy
    Simulated_data.make_treatment(behavior_policy)

    # Retrieve the data 
    data = Simulated_data.get_data()

    # Generate the target policy 
    target_policy, GroundTruth = Simulated_data.generate_target_policy(args.epislon)
    print("Ground Truth:", GroundTruth)
    print(target_policy)

    baseline = {}
    ips = []
    snips = []
    DM_xgBoost = []
    DR_xgBoost = []
    DM_AE = []
    SwitchDR = []
    groupIPS = []
    TrueIPS = []
    groupIPSNoOrder = []
    DR_AE = []
    groupIPSK = {i:[] for i in GROUP_NUM}

    for i in tqdm(range(K)):
        Simulated_data.make_treatment(behavior_policy)
        data = Simulated_data.get_data()

        print("Variance of reward:", data['reward'].var())

        if args.truePropensity:
            p_score = behavior_policy

        else:
            p_score = cal_propensity_score(data['context'], data['action'], device=device, verbose=True)

        regression_model = RegressionModel(
            n_actions=args.t_d,
            # base_model=xgb.XGBRegressor(n_estimators=100,random_state=1,nthread=5),
            base_model=RandomForestRegressor(
                        n_estimators=100,
                        max_samples=0.8,
                        random_state=1,
                        n_jobs=5
                    )
        )

        # regression_MLP = RegressionModel(
        #     n_actions=args.t_d,
        #     base_model=MLPRegressor(random_state=1,
        #                         max_iter=1000),
        # )
        data['position'] = None
        data['pscore'] = p_score[np.arange(data["context"].shape[0]), np.squeeze(data["action"])]
        estimated_rewards = regression_model.fit_predict(
            context=data["context"],  # context; x
            action=data["action"],  # action; a
            reward=data["reward"],  # reward; r
            n_folds=2,
            random_state=12345,
        )

        estimated_rewards_dict = {
            "DM_xgBoost": estimated_rewards,
            "DR_xgBoost": estimated_rewards,
            "SwitchDR": estimated_rewards
        }
        ope = OffPolicyEvaluation(
            bandit_feedback=data,
            ope_estimators=[
                IPS(estimator_name="IPS"),
                SNIPS(estimator_name="SNIPS"),
                DM(estimator_name="DM_xgBoost"),
                DR(estimator_name="DR_xgBoost"),
                Switch(estimator_name="SwitchDR", lambda_=150)
            ]
        )

        estimated_policy_values = ope.estimate_policy_values(
            # ground_truth_policy_value=GroundTruth,
            action_dist=np.expand_dims(target_policy, 2),
            estimated_rewards_by_reg_model=estimated_rewards_dict,
            # metric="se",
        )

        ips.append(estimated_policy_values['IPS'])
        snips.append(estimated_policy_values['SNIPS'])
        DM_xgBoost.append(estimated_policy_values['DM_xgBoost'])
        DR_xgBoost.append(estimated_policy_values['DR_xgBoost'])
        SwitchDR.append(estimated_policy_values['SwitchDR'])

        print(estimated_policy_values)


        # Implementing Gumbel Group
        data = Simulated_data.get_group_data()
        params = {
            'lr': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size}
        embedding_size = args.embedding_size

        import time
        start = time.time()
        model, one_hot_t = run_latent_method(train_x=data['context'],
                                train_t=data['action'],
                                train_y=data['reward'],
                                group_k=args.group_k,
                                ori_k=args.t_d,
                                params=params,
                                device=device,
                                embedding_size=embedding_size
                                )

        dmEstimator = DirectMethod.DirectMethod(context=data['context'],
                                            treatment=data['action'],
                                            reward=data['reward'])
        dm_reward = dmEstimator.estimate_by_ae(reward=data['reward'],
                                            target_policy=target_policy,
                                            model=model,
                                            device=device)
        print("Direct mehthod", dm_reward)
        DM_AE.append(dm_reward)     
        # print("LipMLP - running time:", time.time() - start)
        # p_score = cal_propensity_score(data['context'], data['action'], device=device, verbose=True)
        train_t_torch = torch.from_numpy(data['action']).float().to(device)
        onehot_t = F.one_hot(train_t_torch.long().squeeze()).float()

        if args.filePath == "num_k":
            for k in GROUP_NUM:
                gips = HierachicalGroup(model=model,
                                    train_x=data['context'],
                                    train_t=data['action'],
                                    train_y=data['reward'],
                                    ori_t_k=args.t_d,
                                    group_k=k,
                                    target_policy=target_policy,
                                    device=device,
                                    estimator="IPW",
                                    onehot_t=onehot_t,
                                    slope=False,
                                    raw_pscore=p_score,
                                    true_pscore=args.truePropensity)
                groupIPSK[k].append(gips)
        else:
            gips = HierachicalGroup(model=model,
                                train_x=data['context'],
                                train_t=data['action'],
                                train_y=data['reward'],
                                ori_t_k=args.t_d,
                                group_k=args.group_k,
                                target_policy=target_policy,
                                device=device,
                                estimator="IPW",
                                onehot_t=onehot_t,
                                slope=False,
                                raw_pscore=p_score,
                                true_pscore=args.truePropensity,
                                visual=args.visual)
            if args.visual == True:
                groupVisulisation(None, None, gips, 6, 30)
                return 1
            groupIPS.append(gips)

        
        action_to_bin = Simulated_data.get_action_to_bin()
        gipsTrue = TrueGroupIPS(model=None,
                                pscore=p_score,
                                train_x=data['context'],
                                train_t=data['action'],
                                train_y=data['reward'],
                                ori_t_k=args.t_d,
                                group_k=args.group_k,
                                target_policy=target_policy,
                                device=device,
                                bin_dict=action_to_bin).mean()
        
        # gipsNoOrder = TrueGroupIPS(model=None,
        #                         pscore=p_score,
        #                         train_x=data['context'],
        #                         train_t=data['action'],
        #                         train_y=data['reward'],
        #                         ori_t_k=args.t_d,
        #                         group_k=args.group_k,
        #                         target_policy=target_policy,
        #                         device=device).mean()

        print("Group-IPS:", gips)
        print("Group-IPS(True)", gipsTrue)
        

        TrueIPS.append(gipsTrue)
        # groupIPSNoOrder.append(gipsNoOrder)

    # Baseline
    baseline['IPS'] = ips
    baseline['SNIPS'] = snips
    baseline['DM_AE'] = DM_AE
    baseline['DM_xgBoost'] = DM_xgBoost
    baseline['DR_xgBoost'] = DR_xgBoost
    baseline['SwitchDR'] = SwitchDR



    results = {
        'baseline': baseline,
        'GroupIPS': groupIPS,
        'GroupIPS(True)': TrueIPS,
        # 'GroupIPS(unordered)':groupIPSNoOrder,
        'ground_truth': GroundTruth
    }
    if args.filePath == 'num_k':
        results['GroupIPS'] = groupIPSK
    validate_results(results, Lambda=0, args=args)




def HierachicalGroup(model,
            train_x,
            train_t,
            train_y,
            ori_t_k,
            group_k,
            target_policy,
            device,
            slope=False,
            estimator='IPW',
            onehot_t=None,
            raw_pscore=None,
            true_pscore=False,
            visual=False
            ):
    """
        Define the Euclidean distance based on the action embedding space. 

        Here we define the notation of each parameters:
            - Ori_k: The original space of treatment.
            - Target_k: The target number of treatment after KNN Grouping.
            - Distance Metric: Euclidean or any distance that measures well in embedding space.

        We cluster the distance from bottom to top, so |T| can be efficiently reduced 
        to |Z| treatment where |Z| << |T|.
    """

    train_x_torch = torch.from_numpy(train_x).float().to(device)
    train_t_torch = torch.from_numpy(train_t).float().to(device)
    train_y_torch = torch.from_numpy(train_y).float().to(device)

    # We perform hierachical clustering 
    one_hot =  F.one_hot(torch.Tensor([[i] for i in range(ori_t_k)]).long().squeeze()).float()
    
    # Identify the grouping step
    _, embedding = model(train_x_torch[0:ori_t_k], one_hot.to(device))
    embedding = embedding.detach().cpu().numpy()
    clustering = AgglomerativeClustering(n_clusters=group_k,
                                        affinity='euclidean',
                                        linkage='complete',
                                        distance_threshold=None)
    # clustering = KMeans(n_clusters=group_k, random_state=0)
    clustering = clustering.fit(embedding)
    groups = clustering.labels_
    if visual:
        return groups


    # Convert treatment to the assigned bin
    _, group_t =  model(train_x_torch, onehot_t.to(device))
    group_t = group_t.detach().cpu().numpy().squeeze()

    bin_dict = dict()
    embedding = embedding.squeeze()
    for i in range(embedding.shape[0]):
        bin_dict[i] = groups[i]

    train_t_copy = copy(train_t)
    train_t_copy = train_t_copy.squeeze()
    for i in range(group_t.shape[0]):
        train_t_copy[i] = bin_dict[train_t_copy[i]]

    
    print("Bin:", list(bin_dict.values()))
    train_t_copy = train_t_copy.astype(int)
    # Convert the target policy
    group_target_policy = np.zeros(shape=(train_x.shape[0], group_k))
    for i in range(train_x.shape[0]):
        for j in range(ori_t_k):
            group = bin_dict[j]
            group_target_policy[i][group] += target_policy[i][j] 

    if true_pscore:
        p_score = np.zeros(shape=(train_x.shape[0], group_k))
        for i in range(train_x.shape[0]):
            for j in range(ori_t_k):
                group = bin_dict[j]
                p_score[i][group] += raw_pscore[i][j] 
    else:
        p_score = cal_propensity_score(train_x, train_t_copy, device=device, verbose=True)

    if estimator == "IPW":
        ipwEstimator = IPW.IPW(context=train_x,
                               treatment=train_t_copy,
                               reward=train_y)

        ipw_reward = ipwEstimator.estimate_reward(reward=train_y,
                                                action=train_t_copy,
                                                pscore=p_score,
                                                target_policy=group_target_policy)
        if slope:
            ipw_cnf = np.sqrt(np.var(ipw_reward) / (train_x.shape[0] - 1))
            delta = 0.05
            ipw_cnf *= stats.t.ppf(1.0 - (delta / 2), train_x.shape[0])
            
            print(ipw_reward.mean(), ipw_cnf, group_k)
            return ipw_reward.mean(), ipw_cnf
        else:
            return ipw_reward.mean()

    elif estimator == "SNIPW":
        snipwEstimator = SNIPW.SNIPW(context=train_x,
                               treatment=train_t_copy,
                               reward=train_y)
        snipw_reward = snipwEstimator.estimate_reward(reward=train_y,
                                                    action=train_t_copy,
                                                    pscore=p_score,
                                                    target_policy=group_target_policy)
        if slope:
            snipw_cnf = np.sqrt(np.var(snipw_reward) / (train_x.shape[0] - 1))
            delta = 0.05
            snipw_cnf *= stats.t.ppf(1.0 - (delta / 2), train_x.shape[0])
            return snipw_reward.mean(), snipw_cnf
        else:
            return snipw_reward.mean()

    elif estimator == "DR":

        drEstimator = DoublyRobust.DoublyRobust(context=train_x,
                                                reward=train_y)
        dr_reward = drEstimator.estimate_reward(reward=train_y,
                                                action=train_t,
                                                pscore=raw_pscore,
                                                target_policy=target_policy,
                                                group_target_policy=group_target_policy,
                                                group_action=train_t_copy,
                                                onehot_t=onehot_t,
                                                group_pscore=p_score,
                                                device=device,
                                                model=model)

        if slope:
            dr_cnf = np.sqrt(np.var(dr_reward) / (train_x.shape[0] - 1))
            delta = 0.05
            dr_cnf *= stats.t.ppf(1.0 - (delta / 2), train_x.shape[0])
            return dr_reward.mean(), dr_cnf
        else:
            return dr_reward.mean()



def slopeFineTune(model,
            train_x,
            train_t,
            train_y,
            ori_t_k,
            group_k,
            target_policy,
            device,
            estimator='IPW',
            onehot_t=None,
            raw_pscore=None,
            true_pscore=False,
            squared_errors=None,
            policy_value=None,
            policy_list=None
            ):
    """
        SLOPE++ method proposed by Tucker and Lee.(2021)

        It is a data driven approach for selecting the best hyperparameter group.
        It measures the bias and variance with some reasonable assumptions.
        We selects the best number of groups using SLOPE++ method.
    """
    C = np.sqrt(6) - 1
    theta_ipw_list, cnf_ipw_list = [], []
    theta_ipw_for_sort, cnf_ipw_for_sort = [], []
    ipw_reward = None

    K = [2,3,5,10,20,50,80,100]
    # It computes the bias and variance from |T| to min number of |Z|
    for i in tqdm(K):
        theta_i, cnf_i = HierachicalGroup(model=model,
                                        train_x=train_x,
                                        train_t=train_t,
                                        train_y=train_y,
                                        ori_t_k=ori_t_k,
                                        group_k=i,
                                        target_policy=target_policy,
                                        device=device,
                                        slope=True,
                                        estimator=estimator,
                                        onehot_t=onehot_t,
                                        raw_pscore=raw_pscore,
                                        true_pscore=true_pscore)
        squared_errors['G-{}, k={}'.format(estimator, i)] = (theta_i - policy_value) ** 2
        policy_list['G-{}, k={}'.format(estimator, i)] = theta_i
        if len(theta_ipw_list) > 0:
            theta_ipw_for_sort.append(theta_i), cnf_ipw_for_sort.append(cnf_i)         
        else:
            theta_ipw_list.append(theta_i), cnf_ipw_list.append(cnf_i)

    idx_list = np.argsort(cnf_ipw_for_sort)[::-1]

    count = 0
    for idx in idx_list:
        theta_i, cnf_i = theta_ipw_for_sort[idx], cnf_ipw_for_sort[idx]
        theta_j, cnf_j = np.array(theta_ipw_list), np.array(cnf_ipw_list)   

        if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
            print("-"*100)
            print(theta_j, theta_i, cnf_i, cnf_j,C)
            theta_ipw_list.append(theta_i), cnf_ipw_list.append(cnf_i)
            count += 1
        else:
            ipw_reward = theta_j[-1]
            break
    if ipw_reward is None:
            ipw_reward = theta_j[-1]
        
    print("Optimal dimension for {} is {}".format(estimator, ori_t_k - count))
    print("Group-{}:{}".format(estimator, ipw_reward))

    return ipw_reward



def TrueGroupIPS(model,
            pscore,  
            train_x,
            train_t,
            train_y,
            ori_t_k,
            group_k,
            target_policy,
            device,
            bin_dict=None):
    # group_k = int(ori_t_k//5)
    if bin_dict == None:
        action_in_group = int(ori_t_k//group_k)
        bin_dict = {i:int(i//action_in_group) for i in range(ori_t_k)}
    else:
        group_k = group_k


    p_score = np.zeros(shape=(train_x.shape[0], group_k))
    for i in range(train_x.shape[0]):
        for j in range(ori_t_k):
            group = bin_dict[j]
            p_score[i][group] += pscore[i][j]
            

    for i in range(train_x.shape[0]):
        train_t[i] = bin_dict[train_t[i][0]]
    
    # Convert the target policy
    group_target_policy = np.zeros(shape=(train_x.shape[0], group_k))
    for i in range(train_x.shape[0]):
        for j in range(ori_t_k):
            group_target_policy[i][bin_dict[j]] += target_policy[i][j] 



    ipwEstimator = IPW.IPW(context=train_x,
                               treatment=train_t,
                               reward=train_y)
    
    ipw_reward = ipwEstimator.estimate_reward(reward=train_y,
                                                  action=train_t,
                                                  pscore=p_score,
                                                  target_policy=group_target_policy)
    
    return ipw_reward

if __name__ == '__main__':
    main()
