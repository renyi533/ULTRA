
import sys
name = sys.argv[1]
print(name)
for step in range(2000, 12000, 2000):
    filename = "logs/mtl_%s_clicksimu.log.%s" %(name, step)
    with open(filename) as f:
        lines = f.readlines()
        metrics = lines[-1].strip().split(' ', 1)[1]
        print(step, metrics)

