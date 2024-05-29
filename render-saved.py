import argparse
import json

from agent import ACAgent, SNACAgent, SEACAgent
from environments import RwareEnvironment, ForagingEnvironment, PettingZooEnvironment

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="waterworld")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--algo", type=str, default="SEAC", choices=["IAC", "SNAC", "SEAC", "ISAC"])
    argparser.add_argument("--episode_max_length", type=int, default=200)

    args = argparser.parse_args()

    env_name = args.env
    episode_max_length = args.episode_max_length
    is_discrete = False

    if "rware" in env_name.lower():
        env = RwareEnvironment(env_name=env_name, max_steps=episode_max_length, render=True)
        is_discrete = True
    elif "foraging" in env_name.lower():
        env = ForagingEnvironment(env_name=env_name, max_steps=episode_max_length, render=True)
        is_discrete = True
    elif "multiwalker" in env_name.lower():
        env = PettingZooEnvironment(env_name="multiwalker", max_steps=episode_max_length, render=True)
    elif "waterworld" in env_name.lower():
        env = PettingZooEnvironment(env_name="waterworld", max_steps=episode_max_length, render=True)
    else:
        raise NotImplementedError

    env.reset(seed=args.seed)

    agent_type = args.algo
    agent_list = []
    agent_args = json.load(open(f"logs/{args.env}/{args.algo}/5/{args.seed}/args.json"))
    print(agent_args)

    # Individual agents with no access to each other
    if agent_type == "IAC":
        for agent_id in env.agents:
            agent = ACAgent(env.observation_shapes[agent_id], env.action_shapes[agent_id],
                            capacity=agent_args["buffer_size"], device=agent_args["device"],
                            batch_size=agent_args["batch_size"],
                            n_steps=agent_args["n_steps"], is_discrete=is_discrete)
            agent_list.append(agent)

    # Agents use a single actor network, and the master calculates loss from all memories
    elif agent_type == "SNAC":
        num_agents = len(env.agents)
        master = SNACAgent(env.observation_shapes[env.agents[0]], env.action_shapes[env.agents[0]],
                           capacity=agent_args["buffer_size"] * num_agents, device=agent_args["device"], master=None,
                           agent_list=agent_list,
                           batch_size=agent_args["batch_size"], n_steps=agent_args["n_steps"], is_discrete=is_discrete)
        agent_list.append(master)
        for i in range(1, num_agents):
            agent_id = env.agents[i]
            agent = SNACAgent(env.observation_shapes[agent_id], env.action_shapes[agent_id],
                              capacity=agent_args["buffer_size"], device=agent_args["device"], master=master,
                              agent_list=agent_list,
                              batch_size=agent_args["batch_size"], n_steps=agent_args["n_steps"],
                              is_discrete=is_discrete)
            agent_list.append(agent)
    elif agent_type == "SEAC":
        for agent_id in env.agents:
            agent = SEACAgent(env.observation_shapes[agent_id], env.action_shapes[agent_id],
                              capacity=agent_args["buffer_size"], device=agent_args["device"],
                              agent_list=agent_list, lambda_value=agent_args["SEAC_lambda_value"],
                              batch_size=agent_args["batch_size"],
                              n_steps=agent_args["n_steps"], is_discrete=is_discrete)
            agent_list.append(agent)

    for agent_id, agent in enumerate(agent_list):
        agent.load(f"logs/{args.env}/{args.algo}/5/{args.seed}/agent_{agent_id}.pth")

    while True:
        states, info = env.reset()
        while env.agents:
            actions = {agent_id: agent.act(states[agent_id], training=True)
                       for agent, agent_id in zip(agent_list, env.possible_agents)}

            states, _, _, _, _ = env.step(list(actions.values()))
