import os
import dump_to_bvh


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("in_dir", type = str)
parser.add_argument("out_dir", type = str)

args = parser.parse_args()

default_channels = [dump_to_bvh.channel_order[c] for c in 'ZYX']

for path, dirnames, filenames in os.walk(args.in_dir):
    for f in filenames:
        if (f[-4:] == '.txt'):
            subject = path[path.rindex('/') + 1:]
            inputfile = os.path.join(path, f)
            outputfile = os.path.join(args.out_dir, "%s_%s.bvh"%(subject, f[:-4]))
            

            print("Found %s, dumping to %s"%(inputfile, outputfile))
            dump_to_bvh.main(inputfile, outputfile, default_channels, fps = 50, noroot = True)



