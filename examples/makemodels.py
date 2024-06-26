import argparse
import json
from subprocess import call

path = "./data/models"

coords = {}

for i, c in enumerate(['x', 'y', 'z']):
    base = [0, 0, 0]
    base[i] = 1
    coords [c] = base.copy()

    base[i] = -1
    coords['-' + c] = base.copy()



parser = argparse.ArgumentParser()
parser.add_argument('--h36m', action = 'store_true')
parser.add_argument('infile', type = str)
parser.add_argument('joint', type = str)

args = parser.parse_args()

if (args.h36m):
    outcust_template = 'h36m_out_%s%s'
    outfile_template = './data/models/h36m_out_%s%s.json'
    tposefile = './data/target/h36m_S1_tpose.bvh'
    outputdir = './output_h36mtest'
else:
    outcust_template = 'zed34_out_%s%s'
    outfile_template = './data/models/zed34_out_%s%s.json'
    tposefile = './data/target/zed_34_tpose_flipped.bvh'
    outputdir = './output_zedtest'
    
with open(args.infile, 'r') as fp:
    js = json.load(fp)

    for xv in sorted(coords):
        for yv in sorted(coords):            
            js['model']['cos_map'][args.joint]['x'] = coords[xv]
            js['model']['cos_map'][args.joint]['y'] = coords[yv]
            with open(outfile_template%(xv, yv), 'w') as ofp:
                json.dump(js, ofp, indent = 4)
            
        
for xv in sorted(coords):
    for yv in sorted(coords):            
        call(['python',
            './run_retargeting.py',
              tposefile,
              outcust_template%(xv, yv),
              'data/testsrc',
              'custom',
              outputdir])
