import argparse
import os
import torch
import json

from experimenter import create_experiment


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="rware-tiny-4ag-v1")
    argparser.add_argument("--num_agents", type=int, default=4) # TODO: this shouldn't be hardcoded in the future (e.g. for different envs)
    argparser.add_argument("--agent_type", type=str, default="SEAC", choices=["IAC", "SNAC", "SEAC", "SESAC"])
    argparser.add_argument("--episode_max_length", type=int, default=500) # TODO: this shouldn't be hardcoded in the future (e.g. for different envs)
    argparser.add_argument("--total_env_steps", type=int, default=50000000)
    argparser.add_argument("--warmup_episodes", type=int, default=0)
    argparser.add_argument("--pretrain_path", type=str, default=None)
    argparser.add_argument("--save_path", type=str, default="logs/")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--evaluate_frequency", type=int, default=100)
    argparser.add_argument("--evaluate_episodes", type=int, default=5)
    
    argparser.add_argument("--num_gradient_steps", type=int, default=5)
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    
    argparser.add_argument("--render", default=False, action="store_true")
    
    argparser.add_argument("--n_steps", type=int, default=5)
    
    
    # SEAC related
    argparser.add_argument("--SEAC_lambda_value", type=float, default=1.0)
    
    args = argparser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = "cpu" # GPU overhead is greater than speedup gains
    # TODO: delete
    # args.render = True
    
    args.save_path = f"{args.save_path}/{args.env}/{args.agent_type}/{args.n_steps}/{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    # Save arguments
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    experimenter = create_experiment(args)
    experimenter.run(args)
    
    
    
    
    