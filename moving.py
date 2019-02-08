import os
fname = "white.txt"
with open(fname) as f:
    for line in f:
        current = line.replace(" ", "").replace("\n", "")
        os.rename(current, "Normal/" + current)
