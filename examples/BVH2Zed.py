
# Turns a BVH file with a ZED-formatted skeleton into a viable keypoints / rotations list close to the Dancegraph signal format

import numpy as np
import argparse

# These are the 18 standard bones. How do we fill in the keypoints for the other 14?
bone_list_34 = {
    "Pelvis" : 0,
    "NavalSpine" : 1,
    "ChestSpine" : 2,
    "Neck" : 3,
    "LeftClavicle" : 4,
    "LeftShoulder" : 5,
    "LeftElbow" : 6,
    "LeftWrist" : 7,

    "RightClavicle" : 11,
    "RightShoulder" : 12,
    "RightElbow" : 13,
    "RightWrist" : 14,

    "RightHip" : 22,
    "RightKnee" : 23,
    "RightAnkle" : 24,
    "LeftHip" : 18,
    "LeftKnee" : 19,
    "LeftAnkle" : 20
    }




    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type = str)
    args = parser.parse_args()

