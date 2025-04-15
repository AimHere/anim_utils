
# Turns a BVH file into a csv file with keypoint positions

import numpy as np
import argparse
import re

import json
import struct

from bvh import Bvh

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import math

from scipy.spatial.transform import Rotation

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi

class ForwardKinematics:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpos = Position([0,0,0])):
        self.bonetree = bonetree
        self.bonelist = bonelist
        self.root = rootbone
        self.tpose = [Position(p) for p in tpose]

        
    def propagate(self, rotations, initial_position):

        keyvector = [Position([0, 0, 0]) for i in range(34)]
        
        def _recurse(bone, c_rot, pIdx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                n_rot = c_rot
                new_pos = initial_position
            else:
                n_rot = c_rot * rotations[pIdx]
                new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])

            keyvector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return keyvector

class BVHReader():
    
    def __init__(self, filename):
        with open(self.filename, 'r') as fp:
            self.mocap = Bvh(fp.read())
            self.eulers = self.get_rotations()

    def get_quaternions(self, frame, quantized = False):

        rotations = [Quaternion.zero() for i in range(34)]

        # Weird feet hack. Why is this necessary?
        rotations[body_parts34.index('LEFTHEEL')] = Quaternion([math.sqrt(2), 0, 0, math.sqrt(2)])
        rotations[body_parts34.index('RIGHTHEEL')] = Quaternion([math.sqrt(2), 0, 0, math.sqrt(2)])                  
        
        for joint in self.mocap.get_joints_names():
            jidx = body_parts34.index(joint.upper())
            chan = self.mocap.joint_channels(joint)

            perm = "".join([c[0].lower() for c in chan if c[1:] == 'rotation'])
            rfval = self.mocap.frame_joint_channels(frame, joint, chan)

            zindex = perm.index('z')
            fval = rfval.copy()
            fval[zindex] = -fval[zindex]

            if (quantized):
                rotations[jidx] = Euler(fval, perm = perm).toQuantQuat(xneg = True, zneg = False)                
                if (self.noroot):
                    rotations[0] = Quantized_Quaternion.zero()
            else:
                rotations[jidx] = Euler(fval, perm = perm).toQuat(xneg = True, zneg = False)
                if (self.noroot):
                    rotations[0] = Quaternion.zero()

        return rotations

    def get_keypoints(self):
        fk = ForwardKinematics(self.body_parts, self.body_tree, self.root_bone, self.tpose)
        kp_ = [fk.propagate(self.get_quaternions(frame), Position([0, 0, 0])) for frame in range(self.mocap.nframes)]

        kp_ = [[Position([k.x, k.y, k.z]) for k in l] for l in kp_]
        return kp

class AnimatedScatterKP:
    def __init__(self, keypoints, skellines = False, savefile = None, fps = 50):

        self.numpoints = len(keypoints[0])

        self.savefile = savefile
        self.data = keypoints

        self.lines = []

        self.fps = fps
        
        self.skellines = skellines
        t = np.array([np.ones(self.numpoints) * i for i in range(len(self.data))]).flatten()
        x = np.array([frame[i].x for frame in self.data for i in range(self.numpoints)])
        y = np.array([frame[i].y for frame in self.data for i in range(self.numpoints)])
        z = np.array([frame[i].z for frame in self.data for i in range(self.numpoints)])

        self.df = pd.DataFrame({'time' : t,
                                'x' : x,
                                'y' : y,
                                'z' : z})
        
        self.fig, self.ax = plt.subplots(1, subplot_kw = dict(projection='3d'))
        self.graph = self.ax.scatter(self.df.x, self.df.y, self.df.z)
        if (self.skellines):
            self.drawlines(self.df.x, self.df.y, self.df.z)

        
        self.ax.view_init(elev=90, azim=00, roll=90)
        self.ax.view_init(elev=90, azim=00, roll=90)

        
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=len(self.data),
                                           interval = 1000.0 / self.fps )

        self.ax.set_xlim(-1000, 1000)
        self.ax.set_ylim(-1000, 1000)
        self.ax.set_zlim(-1000, 1000)        
        
        if (self.savefile is not None):        
            writer = animation.FFMpegWriter(fps = self.fps)
            self.ani.save(self.savefile, writer = writer)
            
        plt.show()
        
    def drawlines(self, dfx, dfy, dfz):
        linex = []
        liney = []
        linez = []

        for b in ZED_34_PARENTS:
            if (ZED_34_PARENTS[b] >= 0):
                linex.append([dfx[b], dfx[ZED_34_PARENTS[b]]])
                liney.append([dfy[b], dfy[ZED_34_PARENTS[b]]])
                linez.append([dfz[b], dfz[ZED_34_PARENTS[b]]])                    
                    
        for i in range(33):
            self.lines.append(self.ax.plot(linex[i], liney[i], linez[i]))
            
    def update_plot(self, num):
        data = self.df[self.df['time'] == num]

        if (self.skellines):
            linex = []
            liney = []
            linez = []
            for b in ZED_34_PARENTS:
                if (ZED_34_PARENTS[b] >= 0):
                    qb = 34 * num + b
                    qp = 34 * num + ZED_34_PARENTS[b]
                    
                    linex.append([data.x[qb], data.x[qp]])
                    liney.append([data.y[qb], data.y[qp]])
                    linez.append([data.z[qb], data.z[qp]])
            for i in range(33):
                self.lines[i][0].set_data_3d(linex[i], liney[i], linez[i])
                    
        self.graph._offsets3d = (data.x, data.y, data.z)
    
parser = argparse.argumentParser()

parser.add_argument("--view", action = 'store_true', help = "Optionally view the file")
parser.add_argument("inputfile", type = str, help = "BVH file to input")
parser.add_argument("outputfile", type = str, help = "CSV output file")

bvh = BVHReader(args.inputfile)

kp = bvh.get_keypoints()
if(args.view):
    anim = AnimatedScatterKP(kp, skellines = True, fps = 1.0 / bvh.mocap.frame_time)

print("Dumping Keypoints to be done here")
