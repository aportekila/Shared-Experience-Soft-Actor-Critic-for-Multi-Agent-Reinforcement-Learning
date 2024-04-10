import argparse
import os
import torch

from experimenter import create_experiment


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="rware-small-4ag-v1")
    argparser.add_argument("--num_agents", type=int, default=4)
    argparser.add_argument("--agent_type", type=str, default="SEAC", choices=["IAC", "SNAC", "SEAC", "SESAC"])
    argparser.add_argument("--episode_max_length", type=int, default=500)
    argparser.add_argument("--capacity", type=int, default=5000)
    argparser.add_argument("--total_env_steps", type=int, default=50000000)
    argparser.add_argument("--warmup_episodes", type=int, default=0)
    argparser.add_argument("--pretrain_path", type=str, default=None)
    argparser.add_argument("--save_path", type=str, default="logs/")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--evaluate_frequency", type=int, default=10)
    argparser.add_argument("--evaluate_episodes", type=int, default=5)
    
    argparser.add_argument("--num_gradient_steps", type=int, default=10)
    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    
    argparser.add_argument("--render", default=False, action="store_true")
    
    
    # SEAC related
    argparser.add_argument("--SEAC_lambda_value", type=float, default=1.0)
    
    args = argparser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = "cpu" # GPU overhead is greater than speedup gains
    # TODO: delete
    # args.render = True
    
    args.save_path = f"{args.save_path}/{args.env}/{args.agent_type}/{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    
    experimenter = create_experiment(args)
    experimenter.run(args)
    
    
    
    
    