import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box
import numpy as np
import traci
import os
import sumolib

class TrafficEnv(MultiAgentEnv):
    def __init__(self, sumo_cfg_path, use_gui=False, max_steps=100):
        self.sumo_cfg = sumo_cfg_path
        self.use_gui = use_gui
        self.binary = "sumo-gui" if use_gui else "sumo"
        self.tl_ids = []
        self.max_steps = max_steps
        self.step_counter = 0

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=100.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()
        traci.start([self.binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"])

        self.step_counter = 0
        self.tl_ids = traci.trafficlight.getIDList()

        obs = {tl: self._get_obs(tl) for tl in self.tl_ids}
        info = {tl: {} for tl in self.tl_ids}
        print("\n⚠️ RESET DEBUG:", {k: (v.shape, v.dtype, type(v)) for k, v in obs.items()})
        return obs, info

    def step(self, actions):
        for tl_id, action in actions.items():
            if action[0] > 0.5:
                current_phase = traci.trafficlight.getPhase(tl_id)
                num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                traci.trafficlight.setPhase(tl_id, (current_phase + 1) % num_phases)

        traci.simulationStep()
        self.step_counter += 1

        obs = {tl: self._get_obs(tl) for tl in self.tl_ids}
        rewards = {tl: -self._get_waiting_time(tl) for tl in self.tl_ids}

        done = self.step_counter >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
        terminated = {tl: done for tl in self.tl_ids}
        terminated["__all__"] = done

        truncated = {tl: False for tl in self.tl_ids}
        truncated["__all__"] = False

        infos = {tl: {} for tl in self.tl_ids}

        return obs, rewards, terminated, truncated, infos

    def _get_obs(self, tl_id):
        lane_ids = traci.trafficlight.getControlledLanes(tl_id)
        obs = np.zeros(4, dtype=np.float32)
        for i, lane in enumerate(lane_ids[:4]):
            value = traci.lane.getLastStepHaltingNumber(lane)
            # Ensure value fits the Box space range and float32 type
            obs[i] = np.clip(np.float32(value), self.observation_space.low[i], self.observation_space.high[i])
        return obs

    def _get_waiting_time(self, tl_id):
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        return sum(traci.lane.getWaitingTime(lane) for lane in lanes[:4])

    def close(self):
        if traci.isLoaded():
            traci.close()
