
# Turns a BVH file with a ZED-formatted skeleton into a viable keypoints / rotations list close to the Dancegraph signal format

import numpy as np
import argparse
import re
from bvh import Bvh

from scipy.spatial.transform import Rotation
body_parts34 = [
    "Pelvis",
    "NavalSpine",
    "ChestSpine",
    "Neck",
    "LeftClavicle",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "LeftHand",
    "LeftHandtip",
    "LeftThumb",
    "RightClavicle",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
    "RightHand",
    "RightHandtip",
    "RightThumb",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "LeftFoot",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "RightFoot",
    "Head",
    "Nose",
    "LeftEye",
    "LeftEar",
    "RightEye",
    "RightEar",
    "LeftHeel",
    "RightHeel"
]


body_parts38 = [
    "Pelvis",
    "Spine_1",
    "Spine_2",
    "Spine_3",
    "Neck",
    "Nose",
    "Left_Eye",
    "Right_Eye",
    "Left_Ear",
    "Right_Ear",
    "Left_Clavicle",
    "Right_Clavicle",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hip",
    "Right_Hip",
    "Left_Knee",
    "Right_Knee",
    "Left_Ankle",
    "Right_Ankle",
    "Left_Big_Toe",
    "Right_Big_Toe",
    "Left_Small_Toe",
    "Right_Small_Toe",
    "Left_Heel",
    "Right_Heel",
    "Left_Hand_Thumb_4",
    "Right_Hand_Thumb_4",
    "Left_Hand_Index_1",
    "Right_Hand_Index_1",
    "Left_Hand_Middle_4",
    "Right_Hand_Middle_4",
    "Left_Hand_Pinky_1",
    "Right_Hand_Pinky_1"]

body_34_tree = { 
    "PELVIS": ["NAVAL_SPINE", "LEFT_HIP", "RIGHT_HIP"],
    "NAVAL_SPINE" : ["CHEST_SPINE"],
    "CHEST_SPINE" : ["LEFT_CLAVICLE", "RIGHT_CLAVICLE", "NECK"],

    "LEFT_CLAVICLE" : ["LEFT_SHOULDER"],
    "LEFT_SHOULDER" : ["LEFT_ELBOW"],
    "LEFT_ELBOW" : ["LEFT_WRIST"],
    "LEFT_WRIST" : ["LEFT_HAND", "LEFT_THUMB"],
    "LEFT_HAND" : ["LEFT_HANDTIP"],
     
    "RIGHT_CLAVICLE" : ["RIGHT_SHOULDER"],
    "RIGHT_SHOULDER" : ["RIGHT_ELBOW"],
    "RIGHT_ELBOW" : ["RIGHT_WRIST"],
    "RIGHT_WRIST" : ["RIGHT_HAND", "RIGHT_THUMB"],
    "RIGHT_HAND" : ["RIGHT_HANDTIP"],
     
    "LEFT_HIP" : ["LEFT_KNEE"],
    "LEFT_KNEE" : ["LEFT_ANKLE"],
    "LEFT_ANKLE" : ["LEFT_FOOT", "LEFT_HEEL"],
    "LEFT_HEEL" : ["LEFT_FOOT"],
    
    "RIGHT_HIP" : ["RIGHT_KNEE"],
    "RIGHT_KNEE" : ["RIGHT_ANKLE"],
    "RIGHT_ANKLE" : ["RIGHT_FOOT", "RIGHT_HEEL"],
    "RIGHT_HEEL" : ["RIGHT_FOOT"],

    "NECK" : ["HEAD", "LEFT_EYE", "RIGHT_EYE"],
    "HEAD" : ["NOSE"],
    "LEFT_EYE" : ["LEFT_EAR"],
    "RIGHT_EYE" : ["RIGHT_EAR"]    
    }

body_38_tree = {
    "PELVIS": ["SPINE_1", "LEFT_HIP", "RIGHT_HIP"],
    
    "SPINE_1": ["SPINE_2"],
    "SPINE_2": ["SPINE_3"],
    "SPINE_3": ["NECK", "LEFT_CLAVICLE", "RIGHT_CLAVICLE"],

    "NECK": ["NOSE"],
    "NOSE": ["LEFT_EYE", "RIGHT_EYE"],
    "LEFT_EYE": ["LEFT_EAR"],
    "RIGHT_EYE": ["RIGHT_EAR"],
    
    "LEFT_CLAVICLE": ["LEFT_SHOULDER"],
    "LEFT_SHOULDER": ["LEFT_ELBOW"],
    "LEFT_ELBOW": ["LEFT_WRIST"],
    "LEFT_WRIST": ["LEFT_HAND_THUMB_4",
                   "LEFT_HAND_INDEX_1",
                   "LEFT_HAND_MIDDLE_4",
                   "LEFT_HAND_PINKY_1"],

    "RIGHT_CLAVICLE": ["RIGHT_SHOULDER"],
    "RIGHT_SHOULDER": ["RIGHT_ELBOW"],
    "RIGHT_ELBOW": ["RIGHT_WRIST"],
    "RIGHT_WRIST": ["RIGHT_HAND_THUMB_4",
                   "RIGHT_HAND_INDEX_1",
                   "RIGHT_HAND_MIDDLE_4",
                   "RIGHT_HAND_PINKY_1"],
    
    "LEFT_HIP" : ["LEFT_KNEE"],
    "LEFT_KNEE" : ["LEFT_ANKLE"],
    "LEFT_ANKLE" : ["LEFT_HEEL", "LEFT_BIG_TOE", "LEFT_SMALL_TOE"],
    
    "RIGHT_HIP" : ["RIGHT_KNEE"],
    "RIGHT_KNEE" : ["RIGHT_ANKLE"],
    "RIGHT_ANKLE" : ["RIGHT_HEEL", "RIGHT_BIG_TOE", "RIGHT_SMALL_TOE"],
}



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

    def __str__(self):
        return Quaternion.toQuaternion.__str__()
    
class Quaternion:

    def __init__(self, floats):
        self.rot = Rotation.from_quat(floats)


    def __mul__(self, q):
        return self.rot * q.rot

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
        return Position(self.rot.apply([x.y, x.x, x.z]))

    def toQuantQuat(self):
        q = self.rot.as_quat()
        if (q[3] < 0):
            qq = -q
        else:
            qq = q            
        ints = [round(32767 * f) for f in qq]
        return Quantized_Quaternion(ints)
    
class Euler:
    def __init__(self, floats):

        self.X = floats[0]
        self.Y = floats[1]
        self.Z = floats[2]

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.X, self.Y, self.Z]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.X, self.Y, self.Z]]))

    def toQuat(self, perm = 'xyz'):
        rot = Rotation.from_euler(perm, [self.X, self.Y, self.Z])
        q = rot.as_quat()
        return Quaternion(list(q))

    def toQuantQuat(self, perm = 'xyz'):
        return self.toQuat(perm).toQuantQuat()
        

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

class BVHReader():

    # Reads a BVH file in ZED signal format
    
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as fp:
            self.mocap = Bvh(fp.read())

        self.rots = [get_quaternions(i) for i in range(self.mocap.nframes)]

    def get_quantized_quats(self, frame):

        rotations = [Quaternion.zero() for i in range(34)]
        
        for joint in self.mocap.get_joints_names():
            jidx = body_parts_34.index(joint)
            
            fval = self.mocap.get_frame_joint_channels(frame, joint, ['Xrotation', 'Yrotation', 'Zrotation'])
            
            rotations[jidx] = Euler(fval).toQuantQuat()
        return rotations

    def get_quaternions(self, frame):

        rotations = [Quaternion.zero() for i in range(34)]
        
        for joint in self.mocap.get_joints_names():
            jidx = body_parts_34.index(joint)
            
            fval = self.mocap.get_frame_joint_channels(frame, joint, ['Xrotation', 'Yrotation', 'Zrotation'])
            
            rotations[jidx] = Euler(fval).toQuat()
        return rotations



# These are the 18 standard bones. How do we fill in the keypoints for the other 14?
# bone_list_34 = {
#     "Pelvis" : 0,
#     "NavalSpine" : 1,
#     "ChestSpine" : 2,
#     "Neck" : 3,
#     "LeftClavicle" : 4,
#     "LeftShoulder" : 5,
#     "LeftElbow" : 6,
#     "LeftWrist" : 7,

#     "RightClavicle" : 11,
#     "RightShoulder" : 12,
#     "RightElbow" : 13,
#     "RightWrist" : 14,

#     "RightHip" : 22,
#     "RightKnee" : 23,
#     "RightAnkle" : 24,
#     "LeftHip" : 18,
#     "LeftKnee" : 19,
#     "LeftAnkle" : 20
#     }





if (__name__ == '__main__'):

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type = str)
    args = parser.parse_args()



    
    
