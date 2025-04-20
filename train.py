import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from model.traffic_env import TrafficEnv
import os

# Storage path (with correct URI format)
storage_path = f"file://{os.path.abspath('ray_results')}"

# Register the custom SUMO-based multi-agent environment
def env_creator(config):
    return TrafficEnv(sumo_cfg_path="sumo_config/simulation.sumocfg", use_gui=False)

register_env("traffic_multiagent", env_creator)

# Start local Ray instance
ray.init()

# Create temporary env to extract observation & action spaces
temp_env = env_creator({})

# RLlib PPO Config (with legacy API stack for MultiAgentEnv)
config = (
    PPOConfig()
    .environment(env="traffic_multiagent")
    .framework("torch")
    .env_runners(num_env_runners=1)
    .multi_agent(
        policies={
            "default_policy": (
                None,
                temp_env.observation_space,
                temp_env.action_space,
                {}
            ),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
    )
    .api_stack(  # Important: Disable new API stack for MultiAgentEnv
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

# Run PPO training
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 10},
    storage_path=storage_path,
)
