## **Project: Smart Traffic Lights Using Multi-Agent RL**

### **Goal**
Train multiple agents (traffic lights) to manage intersections in a city grid by learning to minimize congestion, wait time, or emissions.

---

## **1. Tools and Libraries**

### **Simulation Environment**
- **[SUMO (Simulation of Urban Mobility)](https://www.eclipse.dev/sumo/)** – Industry-standard, open-source traffic simulator.
  - Allows you to create real-world or synthetic maps.
  - You can simulate vehicles, traffic lights, pedestrians, etc.
  - Has Python APIs and RL interface.

### **RL Frameworks**
- **[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)** – Scalable, multi-agent reinforcement learning library.
- **Stable-Baselines3** – Simple and effective RL library (mostly for single agent, but can be extended).
- **PettingZoo** – Framework for multi-agent environments (you can wrap your SUMO agents into this for experimentation).
- **OpenAI Gym** – You can convert SUMO into a Gym environment for compatibility.

### **Other Tools**
- **Python** – All modeling, simulation, and training.
- **Matplotlib/Seaborn** – For visualizing traffic trends and RL performance.
- **Jupyter Notebooks or PyCharm** – Development environment.

---

## **2. Dataset Options**

You can either simulate your own dataset or use open traffic data to build realistic scenarios.

### **Synthetic (Recommended for control)**
- Create your own grid city in SUMO.
- Control car spawn rates, road network, traffic lights, etc.
- Useful for controlled experiments.

### **Real-World Datasets**
- **[OpenStreetMap (OSM)](https://www.openstreetmap.org/)** – Export real-world road networks to SUMO format using `netconvert`.
- **[INRIX Traffic Data](https://inrix.com/)** (Paid but high-quality).
- **[Berkeley DeepDrive (BDD100K)](https://bdd-data.berkeley.edu/)** – Some real-world driving data including trajectories.
- **[TLC NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)** – Not for full simulation but great for modeling traffic inflow.

---

## **3. High-Level Steps**

1. **Set up SUMO**
   - Create a grid city or import a map from OSM.
   - Define intersections, signal phases, traffic routes.

2. **Design Your MARL Environment**
   - Each traffic light = agent.
   - State = traffic density in each lane + queue length.
   - Actions = change light phase (green/red/yellow).

3. **Reward Function Design**
   - Minimize wait time.
   - Minimize queue length.
   - Optionally include emissions/fuel usage.

4. **Train MARL Agents**
   - Use RLlib with PPO or DQN for each traffic light.
   - Use centralized or decentralized training approaches.

5. **Evaluate and Visualize**
   - Compare to fixed-timer baseline.
   - Visualize vehicle flows and average wait times.
