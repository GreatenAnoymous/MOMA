import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data
data = pd.read_csv('sample_eval_item1.csv')

# Define the x-axis and y-axes columns
x = data['N']
y_columns = ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'mae']

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Flatten the array of axes for easier iteration
axs = axs.flatten()

# Plot each y-column against the x-axis
for i, col in enumerate(y_columns):
    axs[i].plot(x, data[col], marker='o')
    axs[i].set_title(col)
    axs[i].set_xlabel('N')
    axs[i].set_ylabel(col)
    axs[i].grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('subplot_plot.png')

# Show the plot
plt.show()
