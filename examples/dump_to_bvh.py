import os
import json
from pathlib import Path
import numpy as np
import argparse

from h36m import Human36MReader

from anim_utils.animation_data import BVHReader, MotionVector, SkeletonBuilder   

from anim_utils.animation_data.skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE

from anim_utils.animation_data.constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER

from anim_utils.animation_data.bvh import write_euler_frames_to_bvh_file, convert_quaternion_to_euler_frames

from anim_utils.animation_data import Skeleton

root_channels = ['XRotation', 'YRotation', 'ZRotation']
default_channels = ['XRotation', 'YRotation', 'ZRotation']

MODEL_DATA_PATH = "data" + os.sep + "models"

def create_euler_frame_indices(skeleton):
    nodes_without_endsite = [node for node in list(skeleton.nodes.values()) if node.node_type != SKELETON_NODE_TYPE_END_SITE]
    for node in nodes_without_endsite:
        node.euler_frame_index = nodes_without_endsite.index(node)


def load_json_file(path):
    with open(path, "rt") as in_file:
        return json.load(in_file)

def load_skeleton_model(skeleton_type):
    skeleton_model = dict()
    path = MODEL_DATA_PATH + os.sep+skeleton_type+".json"
    if os.path.isfile(path):
        data = load_json_file(path)
        skeleton_model = data["model"]
    else:
        print("Error: model unknown", path)
    return skeleton_model

def load_motion_from_h36mreader(mv, h36mreader, filter_joints = True, animated_joints = True):
    # H36m is in Exponential map, so needs conversion with both formats
    if (mv.rotation_type == ROTATION_TYPE_QUATERNION):
        quat_frames = h36mreader.get_quaternion_frames()
        mv.frames = quat_frames.reshape([quat_frames.shape[0], -1])
        
    elif (mv.rotation_type == ROTATION_TYPE_EULER):
        euler_frames = h36mreader.get_euler_frames()
        mv.frames = euler_frames.reshape([euler_frames.shape[0], -1])

    mv.n_frames = 0
    mv._prev_n_frames = 0
    mv.frame_time = 1.0 / 60


# Remember to convert ExpMap to Euler Angles
def construct_hierarchy_from_h36m(skeleton, h36m, node_name, level):

    if (node_name == skeleton.root):    
        node = SkeletonRootNode(h36m.tree[node_name][0], root_channels, None, level)
    elif node_name in h36m.tree:

        node = SkeletonJointNode(node_name, default_channels, None, level)
    else:
        node = SkeletonEndSiteNode(node_name, default_channels, None, level)

    nodeidx = h36m.bone_names.index(node_name)
    node.index = nodeidx
    node.offset = list(h36m.offsets[nodeidx, :])

    # Initial Unit Quaternion. From BVH, it looks like it may come from a reference frame
    node.rotation = np.array([1.0, 0.0, 0.0, 0.0])
        
    if node_name in skeleton.animated_joints:
        is_fixed = False
        #quaternion_frame_index
        
    joint_index = -1

    skeleton.nodes[node_name] = node
    if (node_name in h36m.tree):
        for c in h36m.tree[node_name]:
            new_node = construct_hierarchy_from_h36m(skeleton, h36m, c, level + 1)
            new_node.parent = node
            node.children.append(new_node)
    return node


def load_skeleton_from_h36mreader(h36mreader, skeleton_type = None):
    skeleton = Skeleton()
    # Not sure if it's wise to have 'ROOT' as the name of the root bone
    skeleton.root = h36mreader.tree['ROOT'][0]
    nodes = construct_hierarchy_from_h36m(skeleton, h36mreader, h36mreader.tree['ROOT'][0], 0) 
    create_euler_frame_indices(skeleton)
    SkeletonBuilder.set_meta_info(skeleton)
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton

def load_motion_h36m(h36file, skeleton_type = None):
    h36mreader = Human36MReader(h36file)
    mv = MotionVector()
    h36skel = load_skeleton_from_h36mreader(h36mreader, skeleton_type)
    load_motion_from_h36mreader(mv, h36mreader)
    frame_data = np.array(h36mreader.get_euler_frames())
    print(frame_data.shape)
    return h36skel, mv, frame_data
    
def main(infile, outfile):
    p = Path(infile)

    skel, motion, frame_data = load_motion_h36m(infile, "h36m")
    write_euler_frames_to_bvh_file(outfile, skel, frame_data.reshape([frame_data.shape[0], -1]), 1.0/50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run retargeting.')

    parser.add_argument('infile', nargs='?', help='H36M filename')
    #parser.add_argument('skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('output_file', type = str, help = "output file name")

    args = parser.parse_args()

    main(args.infile, args.output_file)
