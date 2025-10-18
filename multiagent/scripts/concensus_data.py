import numpy as np
import matplotlib.pyplot as plt
# Specify the path to your .npy file
file_path = 'D:/IITM/multiagent_opac/multiagent_rl_drone/results/maopac-2025-10-15_17-56-18/rewards.npy'

# Load the data
data = np.load(file_path)

print(f"Data loaded successfully. Shape of data: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"First few elements:\n{data[:20]}")
epochs = [i for i in range(1,len(data)+1)]
plt.plot(epochs, data)
plt.show()