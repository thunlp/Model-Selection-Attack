import numpy as np
import csv
def _read_tsv(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines
lines = _read_tsv("./train_new.tsv")
f=open("index_sst2.txt","r")
idx_list = []
for line in f.readlines():
    idx_list.append(int(line))
f=open("./train_select_new_sst2.tsv","w")
for idx in idx_list:
    f.write("\t".join(lines[idx]))
    f.write("\n")


