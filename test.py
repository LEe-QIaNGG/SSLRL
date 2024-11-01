import numpy as np
num_reward=16
true_reward=[5,10,20,1000]

x = np.arange(len(true_reward))
y = np.array(true_reward)
f = np.interp(np.linspace(0, len(true_reward) - 1, num_reward), x, y)
print(f.tolist())