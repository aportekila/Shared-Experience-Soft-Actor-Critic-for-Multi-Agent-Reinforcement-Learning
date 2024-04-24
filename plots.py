import argparse
import json
import os
from os import listdir
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="Foraging-10x10-3p-3f-v2")
    argparser.add_argument("--smoothing", type=int, default=13)

    args = argparser.parse_args()

    # We will load this from the logs
    total_env_steps = None

    # Figure params
    sns.set(style="darkgrid", font_scale=1.5)
    plt.rcParams['figure.figsize'] = (8, 8)

    for algo in listdir(f"logs/{args.env}"):
        rewards = []
        stds = []
        for seed in listdir(f"logs/{args.env}/{algo}/5/"):  # 5 step TD
            # Load max steps from run args
            if total_env_steps is None:
                seed_args = json.load(open(f"logs/{args.env}/{algo}/5/{seed}/args.json"))
                total_env_steps = seed_args["total_env_steps"]

            # Load history file
            hist_dict = np.load(f"logs/{args.env}/{algo}/5/{seed}/experiment_history.npy", allow_pickle=True).item()
            rewards.append(hist_dict["mean_reward"])
            stds.append(hist_dict["std_reward"])

        # Calc mean over the seeds
        reward_mean = np.mean(np.array(rewards), axis=0)
        # Average SD = sqrt((s1^2 + s2^2 + ... + sk^2)/k)
        reward_std = np.sqrt(np.sum(np.square(np.array(stds)), axis=0) / len(stds))
        time_steps = (np.arange(len(reward_mean)) / len(reward_mean)) * total_env_steps

        # Perform smoothing
        if args.smoothing is not None:
            if args.smoothing < 1 or args.smoothing % 2 == 0:
                raise ValueError(f"Smoothing must be an odd integer greater than 1")
            kernel = np.ones(args.smoothing)
            reward_mean = np.convolve(reward_mean, kernel, mode='same')
            reward_std = np.convolve(reward_std, kernel, mode='same')

        # Plot for this algo
        plt.plot(time_steps, reward_mean, label=algo)
        plt.fill_between(time_steps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)

    # Add labels and title
    plt.xlabel('Environment steps')
    plt.ylabel('Episode return')
    plt.title(f"{args.env} evaluation")
    plt.legend()
    plt.savefig(f"logs/{args.env}_plot.png")
    plt.savefig(f"logs/{args.env}_plot.svg")
    plt.show()
