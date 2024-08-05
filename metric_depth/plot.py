import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
# Read CSV data
data1 = pd.read_csv('sample_LR.csv')
data2 = pd.read_csv('sample_NLRA.csv')
data3 = pd.read_csv('sample_local.csv')

# Define the x-axis and y-axes columns
x = data1['N']
y_columns = ['a1', 'a2',  'abs_rel', 'rmse', 'mae', "runtime"]

label_map = {
    'a1': r'$\delta_{1.05}$',
    'a2': r'$\delta_{1.10}$',
    'abs_rel': 'rel',
    'rmse': 'rmse',
    'mae': 'mae',
    'runtime': 'runtime'
}

# Create subplots
fig, axs = plt.subplots(3, 2)

# Flatten the array of axes for easier iteration
axs = axs.flatten()

# Plot each y-column against the x-axis
for i, col in enumerate(y_columns):
    axs[i].plot(x, data1[col], marker='o')
    axs[i].plot(x, data2[col], marker='D')
    axs[i].plot(data3["N"], data3[col], marker='s')
    # axs[i].set_title(col)
    axs[i].set_xlabel('N')
    if col == "abs_rel":
        #set legend
        print("legend")
        axs[i].legend(["GLRA", "NLRA", "LWLR"])
    if col== "runtime":
        axs[i].set_yscale('log')
    axs[i].set_ylabel(label_map[col])
    # axs[i].grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('samples_plot.pdf',bbox_inches='tight')

# Show the plot
plt.show()
