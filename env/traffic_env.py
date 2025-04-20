import gym
from gym.spaces import Discrete, Box
import numpy as np
import traci
import os
import sumolib

class TrafficEnv(gym.Env):
    def __init__(self, sumo_cfg_path, use_gui=False):
        self.sumo_cfg = sumo_cfg_path
        self.use_gui = use_gui
        self.binary = "sumo-gui" if use_gui else "sumo"
        self.tl_ids = []  # Filled after network loads

        self.action_space = Discrete(2)  # 0 = stay, 1 = switch phase
        self.observation_space = Box(low=0, high=100, shape=(4,), dtype=np.float32)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start([self.binary, "-c", self.sumo_cfg])
        self.tl_ids = traci.trafficlight.getIDList()
        return {tl: self._get_obs(tl) for tl in self.tl_ids}

    def step(self, actions):
        for tl_id, action in actions.items():
            if action == 1:
                traci.trafficlight.setPhase(tl_id, (traci.trafficlight.getPhase(tl_id) + 1) % 4)
        traci.simulationStep()
        obs = {tl: self._get_obs(tl) for tl in self.tl_ids}
        rewards = {tl: -self._get_waiting_time(tl) for tl in self.tl_ids}
        done = {"__all__": traci.simulation.getMinExpectedNumber() == 0}
        return obs, rewards, done, {}

    def _get_obs(self, tl_id):
        # Basic example: use queue lengths or vehicle count
        lane_ids = traci.trafficlight.getControlledLanes(tl_id)
        queue_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in lane_ids[:4]]
        return np.array(queue_lengths, dtype=np.float32)

    def _get_waiting_time(self, tl_id):
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        return sum(traci.lane.getWaitingTime(l) for l in lanes[:4])
