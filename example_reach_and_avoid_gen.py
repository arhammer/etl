import pickle
import os

filenames = []
for dirct in os.listdir("./plan_outputs"):
    fname = "./plan_outputs/" + dirct + "/plan_targets.pkl"
    if os.path.isfile(fname):
        filenames.append(fname)

dump_name = "example_reach_and_avoid.pkl"

g_data = []
reach = 1
for fname in filenames:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    goal = data["obs_g"]
    goal["reach"]=(reach==1)
    g_data.append(goal)
    reach = 1-reach 

with open(filenames[0], "rb") as f:
    alldat = pickle.load(f)

alldat["obs_g"] = g_data

with open(dump_name, "wb") as f:
    pickle.dump(alldat, f)

