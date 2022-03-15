import matplotlib.pyplot as plt
import numpy as np

with open('output/push/log.txt') as f:
    lines = f.readlines()

sample = []
test_return = []
train_return = []
for i, line in enumerate(lines):
    if i > 0:
        x = list(filter(None, line.split(' ')))
        sample.append(float(x[2]))
        test_return.append(float(x[5]))
        train_return.append(float(x[3]))

print(sample)
print(test_return)

plt.plot(sample, test_return)

plt.yticks(np.arange(0, 50, step=5))

plt.grid()
plt.savefig('results.png')