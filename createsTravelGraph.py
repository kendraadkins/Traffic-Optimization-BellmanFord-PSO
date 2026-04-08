import matplotlib.pyplot as plt
import numpy as np

# Data
scenarios = ['Normal Path', 'Construction on 5th', 'Pedestrian Heavy', 'Game Day', 'Police Needed']
bf_times = [128.72, 214.77, 183.47, 418.09, 149.26]  # Bellman-Ford times
pso_times = [135.08, 232.38, 194.62, 442.79, 174.44]  # PSO times
time_savings = [4.89, 8.20, 6.07, 5.92, 16.88]  # Time savings (Bellman-Ford vs PSO)

# Plotting
x = np.arange(len(scenarios))

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the bar chart for average travel times
bar_width = 0.35
ax1.bar(x - bar_width/2, bf_times, bar_width, label='Bellman-Ford', color='white', edgecolor='black', hatch='///')
ax1.bar(x + bar_width/2, pso_times, bar_width, label='PSO',  color='lightgrey', edgecolor='black', hatch='')


# Labeling
ax1.set_xlabel('Scenario')
ax1.set_ylabel('Average Travel Time (seconds)')
ax1.set_title('Comparison of Bellman-Ford and PSO Travel Times')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, rotation=45, ha="right")
ax1.legend()

# Adding time savings annotations
for i, savings in enumerate(time_savings):
    ax1.text(x[i], max(bf_times[i], pso_times[i]) + 10, f'Time Savings: {savings}%', ha='center', va='bottom', color='black')

# Saving the graph
plt.tight_layout()
plt.savefig('averageTravelGraph.png')

plt.show()