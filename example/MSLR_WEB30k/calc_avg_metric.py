import sys
import numpy as np


task = sys.argv[1]
total_step = int(sys.argv[2])
metrics = dict()
for i in range(1, total_step+1):
    filename = "logs/%s_clicksimu%d.log" %(task, i)
    print(filename)
    with open(filename) as f:
        lines = f.readlines()
        words = lines[-1].strip().split(' ')
        for word in words:
            if ":" in word:
                eles = word.split(':')
                if eles[0] not in metrics:
                    metrics[eles[0]] = []
                if eles[1].strip() != "":
                    metrics[eles[0]].append(float(eles[1]))
print("metric", "cnt", "mean", "std")
for key in metrics.keys():
    if len(metrics[key]) == 0:
        continue
    print("%s %d %.4f %.4f" %(key, len(metrics[key]), np.mean(metrics[key]), np.std(metrics[key])))
