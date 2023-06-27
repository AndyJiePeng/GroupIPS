"""
    Utility files for several utility functions.
"""
import math
import pandas as pd
import numpy as np
from syntheticData import Synthetic_data
from model.DirectMethod import treatment_transform


GROUP_NUM = [i + 2 for i in range(49)]
def get_policy(args, context):
    behavior_policy = None
    behavior_policy_type = args.behavior_policy
    if behavior_policy_type == 'uniform':
        # Generating target policy
        behavior_policy = binary_uniform_policy(size=args.size,
                                                t_dimension=args.num_treatments)
    # elif policy_type == 'random':
    #     target_policy = random_policy(size=args.size,
    #                                   t_dimension=args.num_treatments)
    elif behavior_policy_type == 'linear':
        behavior_policy = binary_linear_policy(size=args.size,
                                               t_dimension=args.num_treatments,
                                               context=context)
    return behavior_policy



def validate_results(results, Lambda, args):
    """
        Validate the results for both baseline methods and ours:
        Reported metrics: Bias, SD, MAE, RMSE
        Baseline: IPW, SNIPW, Direct Method, Doubly robust
        Ours:
            - Pure density ratio estimation
            - Latent treatment + density ratio
            - Reconstruction Loss + Latent treatment + density ratio
    """
    # Save several metrics for performance evaluation
    BIAS = []
    SD = []
    MAE = []
    RMSE = []

    groundTruth = results['ground_truth']
    baseline_reward = results['baseline']
    if args.filePath == 'num_k':
        group_ipsK = results['GroupIPS']
    else:
        group_ips = results['GroupIPS']
    group_ipsTrue = results['GroupIPS(True)']
    # group_noOrder = results['GroupIPS(unordered)']
    baseline_method = ['IPS', 'SNIPS', "DM_AE",  'DM_xgBoost', 'DR_xgBoost', 'SwitchDR']
    ours_method = baseline_method
    print("------ Result ------- ")
    print("Ground Truth: [{}]".format(groundTruth))
    for method in baseline_method:
        BIAS.append(calculate_Bias(groundTruth, baseline_reward[method])),
        SD.append(calculate_SD(groundTruth, baseline_reward[method]))
        MAE.append(calculate_MAE(groundTruth, baseline_reward[method]))
        RMSE.append(calculate_RMSE(groundTruth, baseline_reward[method]))
        print('{} - Bias: {:.5f}(SD:{:.5f})  MAE:{:.5f}  RMSE:{:.5f}'.format(
            method,
            calculate_Bias(groundTruth, baseline_reward[method]),
            calculate_SD(groundTruth, baseline_reward[method]),
            calculate_MAE(groundTruth, baseline_reward[method]),
            calculate_RMSE(groundTruth, baseline_reward[method])
        ))

    # Reporting for Density ratio with latent treatment representation
    if args.filePath == 'num_k':
        for k in GROUP_NUM:
            group_ips = group_ipsK[k]
            append_result(BIAS, SD, MAE, RMSE, group_ips, groundTruth)
            ours_method.append('Group IPW(k={})'.format(k))
            print('Group_ips - Bias: {:.5f}(SD:{:.5f})  MAE:{:.5f}  RMSE:{:.5f}'.format(
                calculate_Bias(groundTruth, group_ips).item(),
                calculate_SD(groundTruth, group_ips),
                calculate_MAE(groundTruth, group_ips).item(),
                calculate_RMSE(groundTruth, group_ips)
            ))
        
    else:
        append_result(BIAS, SD, MAE, RMSE, group_ips, groundTruth)
        ours_method.append('Group IPW')
        print('Group_ips - Bias: {:.5f}(SD:{:.5f})  MAE:{:.5f}  RMSE:{:.5f}'.format(
            calculate_Bias(groundTruth, group_ips).item(),
            calculate_SD(groundTruth, group_ips),
            calculate_MAE(groundTruth, group_ips).item(),
            calculate_RMSE(groundTruth, group_ips)
        ))


    append_result(BIAS, SD, MAE, RMSE, group_ipsTrue, groundTruth)
    ours_method.append('GroupIPS(True)')
    print('Group_IPS(True) - Bias: {:.5f}(SD:{:.5f})  MAE:{:.5f}  RMSE:{:.5f}'.format(
        calculate_Bias(groundTruth, group_ipsTrue).item(),
        calculate_SD(groundTruth, group_ipsTrue),
        calculate_MAE(groundTruth, group_ipsTrue).item(),
        calculate_RMSE(groundTruth, group_ipsTrue)
    ))

    # append_result(BIAS, SD, MAE, RMSE, group_noOrder, groundTruth)
    # ours_method.append('GroupIPS(Unordered)')
    # print('Group_IPS(Unordered) - Bias: {:.5f}(SD:{:.5f})  MAE:{:.5f}  RMSE:{:.5f}'.format(
    #     calculate_Bias(groundTruth, group_noOrder).item(),
    #     calculate_SD(groundTruth, group_noOrder),
    #     calculate_MAE(groundTruth, group_noOrder).item(),
    #     calculate_RMSE(groundTruth, group_noOrder)
    # ))
    #
    #
    # Utilizing the pandas dataframe as the output
    # all_methods = baseline_method + ours_method
    all_methods = ours_method
    result_df = pd.DataFrame(
        {'Methods': all_methods,
         'Bias': BIAS,
         'SD': SD,
         'MAE': MAE,
         'RMSE': RMSE
         })

    # Convert the pandas dataframe to csv
    from datetime import datetime

    now = datetime.now()
    file_path = "./groupResult/{}/{}-{}.csv".format(args.filePath, now.month, now.day)
    # Create the directory if not existing

    from csv import writer
    experiment_setting = ['Num_Treatment:{}'.format(args.t_d),
                        'Num_Group:{}'.format(args.group_k),
                        'Size:{}'.format(args.size),
                        'Treatment Effect:{}'.format(args.treatmentEffect),
                        'Beta:{}'.format(args.beta),
                        'UseTruePropensity:{}'.format(args.truePropensity),
                        'STD strength:{}'.format(args.std_strength),
                        "Group number:{}".format(args.vary_group)
                        ]
    with open(file_path, "a") as f:
        f.write('\n')
        writer_object = writer(f)
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(experiment_setting)

    result_df.to_csv(file_path, mode='a', index=False, float_format='%.5f')


def append_result(BIAS, SD, MAE, RMSE, rewards, groundTruth):
    """
        Calculate the metrics for given methods
    """
    BIAS.append(calculate_Bias(groundTruth, rewards).item()),
    SD.append(calculate_SD(groundTruth, rewards))
    MAE.append(calculate_MAE(groundTruth, rewards).item())
    RMSE.append(calculate_RMSE(groundTruth, rewards))


def calculate_Bias(groundTruth, results):
    sum = 0
    for res in results:
        sum += res
    sum /= len(results)
    return abs(sum - groundTruth)


def calculate_SD(groundTruth, results):
    temp = 0
    count = len(results)
    temp_square = 0
    for res in results:
        temp += res
    temp /= count

    for res in results:
        temp_square += (res - temp) ** 2
    temp_square /= count

    return math.sqrt(temp_square)


def calculate_MAE(groundTruth, results):
    sum = 0
    count = len(results)
    for res in results:
        sum += abs(groundTruth - res)
    return sum / count


def calculate_RMSE(groundTruth, results):
    sum = 0
    count = len(results)
    for res in results:
        sum += (groundTruth - res) ** 2

    sum /= count
    return sum


def sample_treatment_reward(Simulated_data,
                            K,
                            behavior_policy,
                            args,
                            sigmoid):
    treatments = []
    rewards = []
    concat_x_list = []
    converted_policy = None
    for i in range(K):
        converted_policy = Simulated_data.make_treatment(policy=behavior_policy,
                                                         policy_type=args.behavior_policy,
                                                         policy_seed=args.seed + i)
        Simulated_data.make_reward(sigmoid=sigmoid,
                                   interaction=args.interaction)
        data = Simulated_data.get_data()

        treatments.append(data['treatment'])
        rewards.append(data['reward'])
        concat_x_list.append(data['data'])

    return treatments, rewards, concat_x_list, converted_policy


def save_model_file(contexts, rewards, args):
    from datetime import datetime

    now = datetime.now()
    # for i in range(len(contexts)):
    path = './syntheticData/T-{}/N-{}Target-{}CS-{}time{}-{}.npz'.format(args.num_treatments,
                                                                              args.size,
                                                                              args.target_policy,
                                                                              args.cs,
                                                                              now.day,
                                                                              now.hour)

    np.savez(path, context=contexts, reward=rewards)



def treatment_effect_estimation():
    """
        Sample K data point with random context information
        See the reward change for each treatment according to the context
    """
    K = 20000
    t_dimension = 5

    Simulated_data = Synthetic_data(size=K,
                                    z_dim=3,
                                    x_dimension=50,
                                    t_dimension=t_dimension,
                                    sparsity=0.3,
                                    cs=0.3,
                                    seed=618)

    context = Simulated_data.make_context()

    treatment_effect = []
    T = np.zeros(shape=(K, t_dimension))
    for i in range(2 ** t_dimension):
        transformed_t = treatment_transform(i, t_dimension)
        for j in range(K):
            T[j] = transformed_t

        Simulated_data.T = T 
        print("Under Treatment:", i)
        Simulated_data.make_reward(interaction=True)
