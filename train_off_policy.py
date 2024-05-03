import argparse
import os
import torch
import json

from experimenter_off_policy import create_of_policy_experiment

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="Foraging-8x8-2p-1f-v2",
                           choices=["multiwalker", "waterwold", "rware-tiny-4ag-v1", "Foraging-10x10-3p-3f-v2", "rware-tiny-4ag-easy-v1"])
    argparser.add_argument("--agent_type", type=str, default="ISAC",
                           choices=["ISAC"])
    argparser.add_argument("--episode_max_length", type=int, default=None)
    argparser.add_argument("--total_env_steps", type=int, default=5_000_000)
    argparser.add_argument("--warmup_episodes", type=int, default=0)
    argparser.add_argument("--pretrain_path", type=str, default=None)
    argparser.add_argument("--save_path", type=str, default="logs/")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--evaluate_frequency", type=int, default=100)
    argparser.add_argument("--evaluate_episodes", type=int, default=5)

    argparser.add_argument("--buffer_size", type=int, default=1_000_000)
    
    argparser.add_argument("--num_gradient_steps", type=int, default=1)
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])

    argparser.add_argument("--render", default=False, action="store_true")



    args = argparser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = "cpu"  # GPU overhead is greater than speedup gains

    args.save_path = f"{args.save_path}/{args.env}/{args.agent_type}/{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    # Save arguments
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    experimenter = create_of_policy_experiment(args)
    experimenter.run(args)
