import sys
import numpy as np


task = sys.argv[1]
total_step = int(sys.argv[2])
if len(sys.argv) > 3:
    log_path = sys.argv[3]
else:
    log_path = "logs"
metrics = dict()
for i in range(1, total_step+1):
    filename = "%s/%s_clicksimu%d.log" %(log_path, task, i)
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
means = dict()
stds = dict()
for key in metrics.keys():
    if len(metrics[key]) == 0:
        continue
    print("%s %d %.4f %.4f" %(key, len(metrics[key]), np.mean(metrics[key]), np.std(metrics[key])))
    means[key] = np.mean(metrics[key])
    stds[key] = np.std(metrics[key])

keys = means.keys()
print(" ".join(keys))
print(" ".join(map(str, ["%.4f" %x for x in means.values()])))
print(" ".join(map(str, ["%.4f" %x for x in stds.values()])))
