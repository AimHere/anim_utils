
# Turns a BVH file with a ZED-formatted skeleton into a viable keypoints / rotations list close to the Dancegraph signal format

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

DANCEGRAPH_SAVE_MARKER = 'DGSAV'
DGSAV_FRAME_MARKER = 1448298308

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi

ZED_34_PARENTS = {}

DGS_HEADER_TEMPLATE = {
    "producer_name" : "camera",
    "save_version" : "0.2",
    "signal_type" : "zed/v2.1",
    "zed_v2.1" : {
        "isReflexive" : True,
        "zedBodySignalType" : "Body_34_KeypointsPlus",
        "zedBufferReadRate" : 30,
        "zedConfidenceThreshold" : 60,
        "zedCoordinateSystem" : "LEFT_HANDED_Y_UP",
        "zedDepth" : "ULTRA",
        "zedFPS" : 60,
        "zedKeypointsThreshold" : 8,
        "zedMaxBodyCount" : 10,
        "zedPlaybackRate" : 1.0,
        "zedRecordVideo" : "",
        "zedResolution" : "720",
        "zedSVOInput" : "",
        "zedSkeletonSmoothing" : 0.7,
        "zedStaticCamera" : True,
        "zedTimeTracking" : False,
        "zedTrackingModel" : "ACCURATE"
        }
}

body_parts34 = [
    "PELVIS",
    "NAVALSPINE",
    "CHESTSPINE",
    "NECK",
    "LEFTCLAVICLE",
    "LEFTSHOULDER",
    "LEFTELBOW",
    "LEFTWRIST",
    "LEFTHAND",
    "LEFTHANDTIP",
    "LEFTTHUMB",
    "RIGHTCLAVICLE",
    "RIGHTSHOULDER",
    "RIGHTELBOW",
    "RIGHTWRIST",
    "RIGHTHAND",
    "RIGHTHANDTIP",
    "RIGHTTHUMB",
    "LEFTHIP",
    "LEFTKNEE",
    "LEFTANKLE",
    "LEFTFOOT",
    "RIGHTHIP",
    "RIGHTKNEE",
    "RIGHTANKLE",
    "RIGHTFOOT",
    "HEAD",
    "NOSE",
    "LEFTEYE",
    "LEFTEAR",
    "RIGHTEYE",
    "RIGHTEAR",
    "LEFTHEEL",
    "RIGHTHEEL"
]


body_parts38 = [
    "PELVIS",
    "SPINE_1",
    "SPINE_2",
    "SPINE_3",
    "NECK",
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_CLAVICLE",
    "RIGHT_CLAVICLE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_BIG_TOE",
    "RIGHT_BIG_TOE",
    "LEFT_SMALL_TOE",
    "RIGHT_SMALL_TOE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_HAND_THUMB_4",
    "RIGHT_HAND_THUMB_4",
    "LEFT_HAND_INDEX_1",
    "RIGHT_HAND_INDEX_1",
    "LEFT_HAND_MIDDLE_4",
    "RIGHT_HAND_MIDDLE_4",
    "LEFT_HAND_PINKY_1",
    "RIGHT_HAND_PINKY_1"]

body_34_tree = { 
    "PELVIS": ["NAVALSPINE", "LEFTHIP", "RIGHTHIP"],
    "NAVALSPINE" : ["CHESTSPINE"],
    "CHESTSPINE" : ["LEFTCLAVICLE", "RIGHTCLAVICLE", "NECK"],

    "LEFTCLAVICLE" : ["LEFTSHOULDER"],
    "LEFTSHOULDER" : ["LEFTELBOW"],
    "LEFTELBOW" : ["LEFTWRIST"],
    "LEFTWRIST" : ["LEFTHAND", "LEFTTHUMB"],
    "LEFTHAND" : ["LEFTHANDTIP"],
     
    "RIGHTCLAVICLE" : ["RIGHTSHOULDER"],
    "RIGHTSHOULDER" : ["RIGHTELBOW"],
    "RIGHTELBOW" : ["RIGHTWRIST"],
    "RIGHTWRIST" : ["RIGHTHAND", "RIGHTTHUMB"],
    "RIGHTHAND" : ["RIGHTHANDTIP"],
     
    "LEFTHIP" : ["LEFTKNEE"],
    "LEFTKNEE" : ["LEFTANKLE"],
    "LEFTANKLE" : ["LEFTFOOT", "LEFTHEEL"],
    "LEFTHEEL" : ["LEFTFOOT"],
    
    "RIGHTHIP" : ["RIGHTKNEE"],
    "RIGHTKNEE" : ["RIGHTANKLE"],
    "RIGHTANKLE" : ["RIGHTFOOT", "RIGHTHEEL"],
    "RIGHTHEEL" : ["RIGHTFOOT"],

    "NECK" : ["HEAD", "LEFTEYE", "RIGHTEYE"],
    "HEAD" : ["NOSE"],
    "LEFTEYE" : ["LEFTEAR"],
    "RIGHTEYE" : ["RIGHTEAR"],

    "LEFTHANDTIP" : [],
    "LEFTTHUMB" : [],
    
    "RIGHTHANDTIP" : [],
    "RIGHTTHUMB" : [],
    
    "NOSE" : [],
    "LEFTEAR" : [],
    "RIGHTEAR" : [],
    
    "LEFTFOOT" : [],
    "RIGHTFOOT" : []

    
    }

body38tree = {
    "PELVIS": ["SPINE1", "LEFTHIP", "RIGHTHIP"],
    
    "SPINE1": ["SPINE2"],
    "SPINE2": ["SPINE3"],
    "SPINE3": ["NECK", "LEFTCLAVICLE", "RIGHTCLAVICLE"],

    "NECK": ["NOSE"],
    "NOSE": ["LEFTEYE", "RIGHTEYE"],
    "LEFTEYE": ["LEFTEAR"],
    "RIGHTEYE": ["RIGHTEAR"],
    
    "LEFTCLAVICLE": ["LEFTSHOULDER"],
    "LEFTSHOULDER": ["LEFTELBOW"],
    "LEFTELBOW": ["LEFTWRIST"],
    "LEFTWRIST": ["LEFTHANDTHUMB4",
                   "LEFTHANDINDEX1",
                   "LEFTHANDMIDDLE4",
                   "LEFTHANDPINKY1"],

    "RIGHTCLAVICLE": ["RIGHTSHOULDER"],
    "RIGHTSHOULDER": ["RIGHTELBOW"],
    "RIGHTELBOW": ["RIGHTWRIST"],
    "RIGHTWRIST": ["RIGHTHANDTHUMB4",
                   "RIGHTHANDINDEX1",
                   "RIGHTHANDMIDDLE4",
                   "RIGHTHANDPINKY1"],
    
    "LEFTHIP" : ["LEFTKNEE"],
    "LEFTKNEE" : ["LEFTANKLE"],
    "LEFTANKLE" : ["LEFTHEEL", "LEFTBIGTOE", "LEFTSMALLTOE"],
    
    "RIGHTHIP" : ["RIGHTKNEE"],
    "RIGHTKNEE" : ["RIGHTANKLE"],
    "RIGHTANKLE" : ["RIGHTHEEL", "RIGHTBIGTOE", "RIGHTSMALLTOE"],

    
}



#    
#    "Hips": [0, 99.672, 0.24705],
#    "Spine": [0, 109.6, -0.98028],
#    "Spine1" : [-0.000138, 121.24, -2.4026],
#    "Spine2" : [-0.000047, 134.6, -4.0291],
#    "Neck" : [-0.000022, 149.63, -3.2362],
#    "Head" : [-0.000019, 159.96, -0.09375],
#    "LeftEye" : [2.9479, 167.64, 9.027],
#    "RightEye" : [-2.9445, 167.64, 9.02767],
#    "LeftShoulder" : [6.1058, 143.71, -3.3235],
#    "LeftArm" : [18.761, 143.45, -5.9245],
#    "LeftForearm" : [46.165, 143.45, -5.9244],
#    "LeftHand" : [73.78, 143.45, -5.9244],
#    "LeftHandMiddle4" : [97.309, 143.45, -5.9244], # LeftHandtip?
#    "LeftHandThumb4" : [87.318, 135.65, 2.7074], #LeftHandThumb
#    "RightShoulder" : [-6.1057, 143.71, -3.3235],
#    "RightArm" : [18.761, 143.45, -5.9244],
#    "RightForearm" : [-46.165, 143.45, -5.9244],
#    "RightHand" : [-73.78, 143.45, -5.9244],
#    "RightHandMiddle4" : [-97.309, 143.45, -5.9244], # RightHandtip?
#    "RightHandThumb4" : [-87.318, 135.65, 2.7074], # RightHandThumb
#    "RightUpLeg" : [-9.1245, 93.016, 0.19167], # RightHip?
#    "RightLeg" : [-9.3691, 52.42, -0.32285],
#    "RightFoot" : [-9.1245, 10.372, -2.3831],
#    "RightToeEnd" : [-9.4978, -0.11983, 20.25],
#    "LeftUpLeg" : [9.1245, 93.016, 0.19167], # LeftHip?
#    "LeftLeg" : [9.3691, 52.42, -0.32537],
#    "LeftFoot" : [9.1245, 10.372, -2.383],
#    "LeftToeEnd" : [9.4978, -0.11993, 20.25],
#    
#    
#    "LeftClavicle"
#    "RightClavicle"
#    "LeftEar"
#    "RightEar"
#    "LeftHeel"
#    "RightHeel"

tpose34_pos = [[0,0,0],
                     [-0.000732270938924443,175.158289701814,0.0000404],
                     [0.104501423707752,350.306093180137,0.061023624087894],
                     [0.209734388920577,525.459141360196,0.122005361332611],
                     [-47.5920764358155,526.439401661188,0.945479045649915],
                     [-173.508163386225,526.509589382348,2.98848444604651],
                     [-413.490495136004,529.070006884027,4.30015451481757],
                     [-644.259234923473,531.557805055099,5.5156119864156],
                     [-690.412981869986,532.055366973068,5.75870846651414],
                     [-782.7204827236,533.050484095638,6.24488492674942],
                     [-737.171470366004,477.177940319642,6.03507728101326],
                     [48.0126939292815,526.382576868867,-0.700813930486266],
                     [173.928779031094,526.312389136708,-2.74381818341968],
                     [413.965281719423,525.913766951398,-5.58751914341834],
                     [644.770225962532,525.03114718918,-8.36370311424063],
                     [690.931216942206,524.854620798345,-8.91894225928869],
                     [783.253196286002,524.501573309859,-10.029413939978],
                     [736.897241876494,469.296359031723,-9.22217198040476],
                     [-97.2538992002065,0,-0.0216466824273884],
                     [-97.2534603765872,-398.665678321646,-0.0280368622539885],
                     [-97.236806268837,-753.034804644565,-0.0408179870540185],
                     [-97.2541801180481,-841.630928132314,106.266711069826],
                     [97.2538982249762,0,0.0216480069085377],
                     [97.2606469820357,-398.664829041876,0.0265545153545871],
                     [97.2769776133806,-753.033961966181,0.0252733646276821],
                     [97.2596464463626,-841.626635633415,106.335670500348],
                     [1.30731388292955,660.085767955841,62.6295143653575],
                     [1.37823874062583,704.938429698234,62.5519366785149],
                     [-25.8996591150869,736.342683178085,31.5073346749533],
                     [-76.4225396149832,715.693774717041,-52.9081045914678],
                     [27.8706194688158,736.259256448681,30.7476818991719],
                     [75.9266183557466,715.457396322932,-55.0604622597071],
                     [-97.2254670188798,-841.625806800333,-35.4809212169046],
                     [97.2881989464546,-841.626116231009,-35.4119624250308]]

# tpose34_pos = [[-a, b, c] for a, b, c in tpose34_pos_xflip]



class Quantized_Quaternion:
    # Represent a quaternion with three 16-bit fixed-point ints
    def __init__(self, ints):
        self.fixed = ints

    def toQuaternion(self):
        floats = [f / 32767 for f in self.fixed]
        sqrs = [f * f for f in floats]
        # print("Sqrs is ", sqrs)
        # print("Sumsq is %f"%(1.0 - sum(sqrs)))
        floats.append(math.sqrt(1.0 - sum(sqrs)))
        return Quaternion(floats)

    def zero():
        return Quantized_Quaternion([0.0, 0.0, 0.0])

    def __str__(self):
        return Quaternion.toQuaternion.__str__()

    def np(self):
        return np.array(self.fixed).astype(np.int16)
    
class Quaternion:

    def __init__(self, floats):
        self.rot = Rotation.from_quat(floats)

    def __mul__(self, q):
        rmul = (self.rot * q.rot).as_quat()
        return Quaternion(rmul)

    def zero():
        return Quaternion([0.0, 0.0, 0.0, 1.0])
        
    def toEuler(self, perm = 'xyz'):
        e = self.rot.as_euler(perm, degrees = True)
        return Euler([e[0], e[1], e[2]])

    def cstr(self, sep = ","):
        q = self.rot.as_quat()
        return (sep.join([str(i) for i in q]))
    
    def __str__(self):
        q = self.rot.as_quat()
        return (" ".join([str(i) for i in q]))

    def apply(self, x):
        return Position(self.rot.apply([x.x, x.y, x.z]))

    def np(self):
        return self.rot.as_quat()
    
    def toQuantQuat(self):
        q = self.rot.as_quat()
        if (q[3] < 0):
            qq = -q
        else:
            qq = q            
        ints = [round(32767 * f) for f in qq]
        return Quantized_Quaternion(ints)
    
class Euler:
    def __init__(self, floats, perm = 'xyz'):

        self.perm = perm
        self.e0 = floats[0]
        self.e1 = floats[1]
        self.e2 = floats[2]

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.e0, self.e1, self.e2]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.e0, self.e1, self.e2]]))

    def toQuat(self, xneg = False, zneg = False):

        rot = Rotation.from_euler(self.perm, [self.e0,
                                              self.e1,
                                              self.e2], degrees = True)
        q = rot.as_quat()

        
        if (xneg == True):
            xcoord = -q[0]
        else:
            xcoord = q[0]
        if (zneg == True):
            zcoord = -q[2]
        else:
            zcoord = q[2]

        return Quaternion([xcoord, q[1], zcoord, q[3]])


    def toQuantQuat(self, xneg = False, zneg = False):
        return self.toQuat(xneg = xneg, zneg = zneg).toQuantQuat()

class Position:
    def __init__(self, floats):
        [self.x, self.y, self.z] = floats

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.x, self.y, self.z]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.x, self.y, self.z]]))


    def scale(self, s):
        return Position([s * self.x , s * self.y, s * self.z])

    def __add__(self, a):
        return Position([self.x + a.x, self.y + a.y, self.z + a.z])

    def __sub__(self, a):
        return Position([self.x - a.x, self.y - a.y, self.z - a.z])

    def __mul__ (self, k):
        return Position([k * self.x, k * self.y, k * self.z])

    
    def np(self):
        return np.array([self.x, self.y, self.z])

    
class Transform:
    def __init__(self, pos, ori):
        self.pos = pos
        self.ori = ori

    def cstrquat(self, sep=","):
        return "%s%s%s"%(self.pos.cstr(sep),sep,self.ori.cstr(sep))
    
    def cstr(self, sep=","):
        return "%s%s%s"%(self.pos.cstr(sep),sep,self.ori.toEuler(args.convert_order).cstr(sep))
    
    def __str__(self):
        return "%s %s"%(self.pos,self.ori.toEuler(args.convert_order))

    def scale(self, x):
        return Transform(self.pos.scale(x), self.ori)

    def offset_pos(self, p):
        return Transform(self.pos + p, self.ori)

class ZedSkelHeader:
    def __init__(self):
        pass

    def read(self, ifp):
        self.numBodies = struct.unpack('b', ifp.read(1))[0]
        self.grab_delta = struct.unpack('i', ifp.read(4))[0]
        self.track_delta = struct.unpack('i', ifp.read(4))[0]
        self.process_delta = struct.unpack('i', ifp.read(4))[0]
        self.padding = struct.unpack('bbb', ifp.read(3))


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


class BVHReader_NP():
    def __init__(self, filename, noroot = False):
        self.filename = filename
        with open(self.filename, 'r') as fp:
            self.mocap = Bvh(fp.read())

        self.noroot = noroot            
        self.eulers = self.get_rotations()
        
    def get_rotations(self):
        self.rots = np.zeros([self.mocap.nframes, 34, 3])
        # self.rots[:, :, 3] = 1.0
        # self.rots[:, body_parts34.index('LEFTHEEL'), 0] = math.sqrt(2)
        # self.rots[:, body_parts34.index('LEFTHEEL'), 3] = math.sqrt(2)
        # self.rots[:, body_parts34.index('RIGHTHEEL'), 0] = math.sqrt(2)
        # self.rots[:, body_parts34.index('RIGHTHEEL'), 3] = math.sqrt(2)


        for frame in range(self.mocap.nframes):
            
            for joint in self.mocap.get_joints_names():
                jidx = body_parts34.index(joint.upper())


                chan = [c for c in self.mocap.joint_channels(joint) if c[1:] == 'rotation']
                
                perm = "".join([c[0].lower() for c in chan if c[1:] == 'rotation'])
                fval = self.mocap.frame_joint_channels(frame, joint, chan)
                zindex = perm.index('z')
                fval[zindex] = -fval[zindex]                
                self.rots[frame, jidx, :] = fval

        
    
    def get_quaternions(self, frame, quantized = False):
        
        rotations = [Quaternion.zero() for i in range(34)]

        # Weird feet hack. Why is this necessary?
        rotations[body_parts34.index('LEFTHEEL')] = Quaternion([math.sqrt(2), 0, 0, -math.sqrt(2)])
        rotations[body_parts34.index('RIGHTHEEL')] = Quaternion([math.sqrt(2), 0, 0, -math.sqrt(2)])                  
        
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

    
class BVHReader():

    # Reads a BVH file in ZED signal format
    
    def __init__(self, filename, noroot = False):
        self.filename = filename
        with open(self.filename, 'r') as fp:
            self.mocap = Bvh(fp.read())

        self.noroot = noroot            
        self.rots = [self.get_quaternions(i) for i in range(self.mocap.nframes)]
        self.skeltype = "Body_34_KeypointsPlus"

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

        # Use the non-quantized rotations
        fk = ForwardKinematics(body_parts34, body_34_tree, 'PELVIS', tpose34_pos)
        print([str(Q) for Q in self.get_quaternions(4)])
        kp_ = [fk.propagate(self.get_quaternions(frame), Position([0, 0, 0])) for frame in range(self.mocap.nframes)]

        kp = [[Position([k.x, k.y, k.z]) for k in l] for l in kp_]
        
        return kp


    def dump_data(self, filename = None):
        if (filename is None):
            if (self.filename [-4:] == '.bvh'):
                new_filename = self.filename[:-4] + ".npz"
            else:
                new_filename = self.filename + ".npz"
        else:
            new_filename = filename
                
        timeval = np.array(self.mocap.frame_time)
        skeltype = np.array(self.skeltype)
        quants = np.array([[c.toQuantQuat().np() for c in r] for r in self.rots])
        keypoints = np.array([[k.np() for k in l] for l in self.get_keypoints()])
        fullquats = np.array([[c.np() for c in r] for r in self.rots])
        np.savez(new_filename, skeltype = skeltype, frametime = timeval, quantized_quats = quants, quats = fullquats, keypoints = keypoints)
    

    def write_header(self, fp):
        fp.write(DANCEGRAPH_SAVE_MARKER.encode('utf-8'))
        jsonbundle = json.dumps(DGS_HEADER_TEMPLATE, indent = 4).encode('utf-8')
        fp.write(struct.pack('i', len(jsonbundle)))
        fp.write(jsonbundle)
        

    def write_frame_metadata(self, fp):
        fp.write(struct.pack('i', DANCEGRAPH_FRAME_MARKER))

        

    # def write_zed_frame_data(self, fp):
    #     pass
    

    # def write_zed_header(self, fp):
    #     pass

    # def write_zedbodies_header(self, fp):
    #     pass

    # def write_zedbodies_body(self, fp):
    #     pass


    
    # def write_frame(self, fp, frame, time):


                 
    
    def write_zed(self, filename):
        
        with open(filename, 'wb') as fp:

            self.write_signal_header(fp)
            time = 0

            for i in range(len(self.rotations)):
                self.write_frame_metadata(fp, time)
                self.write_frame(self, fp, i)
                


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

                
if (__name__ == '__main__'):
    for idx, bone in enumerate(body_parts34):
        try:
            for nbone in body_34_tree[bone]:
                
                nidx = body_parts34.index(nbone)
                ZED_34_PARENTS[nidx] = idx
        except(KeyError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--norootrot", action = 'store_true')
    parser.add_argument("--savefile", type = str, default = None)
    parser.add_argument("infile", type = str)
    args = parser.parse_args()

    bvh = BVHReader(args.infile, noroot = args.norootrot)
    if (args.savefile):
        bvh.dump_data(args.savefile)
    else:
        kp = bvh.get_keypoints()
        anim = AnimatedScatterKP(kp, skellines = True, savefile = args.savefile, fps = 1.0 / bvh.mocap.frame_time)
