import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from env.traffic_env import TrafficEnv
from ray.tune.registry import register_env

def env_creator(env_config):
    return TrafficEnv(sumo_cfg_path="sumo_config/simulation.sumocfg")

if __name__ == "__main__":
    ray.init()
    register_env("traffic_multiagent", lambda config: env_creator(config))

    tune.run("PPO", config={
        "env": "traffic_multiagent",
        "env_config": {},
        "multiagent": {
            "policies": {
                "default_policy": (None, TrafficEnv(None).observation_space, TrafficEnv(None).action_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "default_policy",
        },
        "framework": "torch",
        "num_workers": 1,
    })
