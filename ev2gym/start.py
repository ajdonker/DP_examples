from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
import pickle 
import matplotlib.pyplot as plt
import pandas as pd
config_file = "example_config_files/V2GProfitPlusLoads.yaml"

# Initialize the environment
env = EV2Gym(config_file=config_file,
              save_replay=True,
              save_plots=True)
state, _ = env.reset()
agent = V2GProfitMaxOracle(env,verbose=True) # optimal solution
#        or 
agent = ChargeAsFastAsPossible() # heuristic
for t in range(env.simulation_length):
    actions = agent.get_action(env) # get action from the agent/ algorithm

with open("replay/replay_sim_2025_06_21_018581.pkl", "rb") as f:
    buffer = pickle.load(f)

print(type(buffer))
print(dir(buffer))
# And if there’s a __dict__, dump that too:
if hasattr(buffer, "__dict__"):
    for k,v in buffer.__dict__.items():
        print(k, type(v))


df = pd.DataFrame({
    "time_step": range(buffer.sim_length),
    "total_power": buffer.power_setpoints.sum(axis=1),
    "avg_voltage": buffer.voltages.mean(axis=1),
    "price": buffer.charge_prices
})
df.to_csv("simulation_summary.csv", index=False)