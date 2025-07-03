import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# Change to your actual log paths
dagger_event_path = "/Users/liushijian/Documents/GitHub/homework_fall2023/hw1/data/q2_dagger_ant_Ant-v4_01-07-2025_18-02-00/events.out.tfevents.1751410920.Shijians-MacBook-Pro.local"
bc_event_path = "/Users/liushijian/Documents/GitHub/homework_fall2023/hw1/data/q1_bc_ant_Ant-v4_01-07-2025_18-08-16/events.out.tfevents.1751411296.Shijians-MacBook-Pro.local"

def extract_scalars(event_file, tags=["Eval_AverageReturn", "Eval_StdReturn"]):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    print(f"The tags are: {ea.Tags()}")
    data = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        else:
            data[tag] = []
    return data

bc_data = extract_scalars(bc_event_path)
dagger_data = extract_scalars(dagger_event_path)

# Convert to arrays for plotting
dagger_steps = [step for step, val in dagger_data["Eval_AverageReturn"]]
dagger_returns = [val for step, val in dagger_data["Eval_AverageReturn"]]
dagger_stds = [val for step, val in dagger_data["Eval_StdReturn"]]

# For BC, only one value (usually step 0)
bc_return = bc_data["Eval_AverageReturn"][0][1] if bc_data["Eval_AverageReturn"] else None
bc_std = bc_data["Eval_StdReturn"][0][1] if bc_data["Eval_StdReturn"] else None

# If you want to include expert performance, set the number below:
# expert_return = 4800   # <-- replace with your actual expert mean return

# Plot
plt.figure(figsize=(7,5))
plt.errorbar(dagger_steps, dagger_returns, yerr=dagger_stds, label="DAgger", fmt='-o')
plt.axhline(y=bc_return, color='orange', linestyle='--', label="Behavioral Cloning")
# plt.axhline(y=expert_return, color='green', linestyle='--', label="Expert")

plt.xlabel("DAgger Iteration")
plt.ylabel("Eval Average Return")
plt.title("Ant-v4: Behavioral Cloning vs DAgger Performance")
plt.legend()
plt.tight_layout()
plt.show()