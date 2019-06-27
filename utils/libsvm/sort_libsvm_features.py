import os,sys

input_file = sys.argv[1]
output_file = sys.argv[2]

qid_list_data = {}
with open(input_file) as fin:
    for line in fin:
        arr = line.strip().split(' ')
        qid = int(arr[1].split(':')[1])
        feature_list = arr[2:]
        idx_feature_map = {int(x.split(':')[0]) : x for x in feature_list}
        sorted_idx_feature_list = sorted(idx_feature_map.items(), key=lambda k: k[0])
        if qid not in qid_list_data:
            qid_list_data[qid] = []
        qid_list_data[qid].append(' '.join([arr[0], arr[1]] + [x[1] for x in sorted_idx_feature_list]))

with open(output_file, 'w') as fout:
    sorted_qid_lists = sorted(qid_list_data.items(), key=lambda k: k[0])
    for qid_list in sorted_qid_lists:
        for line in qid_list[1]:
            fout.write(line)
            fout.write('\n')