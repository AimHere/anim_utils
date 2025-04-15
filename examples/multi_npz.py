from subprocess import call
import os
import numpy as np

out_dir = './h36m_zed_npz'
in_dir =  './homeconvert'

for inf in os.listdir(in_dir):
    in_file = os.path.join(in_dir, inf)

    if (inf[-4:] == '.bvh'):
        outf = inf[:-4] + '.npz'
    else:
        outf = inf + '.npz'

    out_file = os.path.join(out_dir, outf)

    commandline = ['python', './BVH2Zed.py', in_file, '--savefile', out_file]
    call(commandline)


