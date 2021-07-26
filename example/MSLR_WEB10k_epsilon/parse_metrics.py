
import sys
task = sys.argv[1]
name = sys.argv[2]
print(name)
for step in range(2000, 12000, 2000):
    filename = "logs/%s_%s_clicksimu.log.%s" %(task, name, step)
    with open(filename) as f:
        lines = f.readlines()
        metrics = lines[-1].strip().split(' ', 1)[1]
        print(step, metrics)

