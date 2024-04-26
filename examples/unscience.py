
# BVH Viewers hate this one weird trick!
# Remove the scientific notation and the weird extra space preceding the 'MOTION' block from the bvh files used by anim_utils, because it breaks bvh viewers

import re
import argparse

def replacement(m):
    f = float(m.string[m.start() : m.end()])
    return "%0.10f"%f

              

pattern = "[+-]*[0123456789.]+e[-+]?[0123456789]+"
def processline(line):
    return re.sub(pattern, replacement, line)


parser = argparse.ArgumentParser()
parser.add_argument("infile", type = str, help = "Input bvh")
parser.add_argument("outfile", type = str, help = "Output bvh")

args = parser.parse_args()

with open(args.infile, "r") as ifp:
    with open(args.outfile, "w") as ofp:
        for iline in ifp.readlines():
            if (len(iline) > 1):
                ofp.write(processline(iline))
