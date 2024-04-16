import json
import os
import argparse
from pathlib import Path
from anim_utils.animation_data import BVHReader, MotionVector, SkeletonBuilder   
from anim_utils.retargeting.analytical import Retargeting, generate_joint_map
from anim_utils.animation_data.bvh import write_euler_frames_to_bvh_file, convert_quaternion_to_euler_frames

from anim_utils.animation_data.skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE

from anim_utils.animation_data.constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER

from anim_utils.animation_data import Skeleton

from h36m import Human36MReader

import numpy as np

MODEL_DATA_PATH = "data"+os.sep+"models"

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

def load_skeleton(path, skeleton_type=None):
    bvh = BVHReader(path)   
    skeleton = SkeletonBuilder().load_from_bvh(bvh)
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton
    

def load_motion_bvh(path, skeleton_type=None):
    bvh = BVHReader(path)   
    mv = MotionVector()  
    mv.from_bvh_reader(bvh)
    print("Shape is ", bvh.frames.shape)
    skeleton = SkeletonBuilder().load_from_bvh(bvh) 
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton, mv

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
    return h36skel, mv
    

root_channels = ['XRotation', 'YRotation', 'ZRotation']

channel_order = {'X' : 'XRotation',
                 'Y' : 'YRotation',
                 'Z' : 'ZRotation' }
#default_channels = ['XRotation', 'YRotation', 'ZRotation']

def create_euler_frame_indices(skeleton):
    nodes_without_endsite = [node for node in list(skeleton.nodes.values()) if node.node_type != SKELETON_NODE_TYPE_END_SITE]
    for node in nodes_without_endsite:
        node.euler_frame_index = nodes_without_endsite.index(node)



def main(src_motion_dir, src_skeleton_type, dest_skeleton, dest_skeleton_type, out_dir, auto_scale=False, place_on_ground=False):
    dest_skeleton = load_skeleton(dest_skeleton, dest_skeleton_type)

    min_p, max_p = dest_skeleton.get_bounding_box()
    quidest_height = (max_p[1] - min_p[1])
    os.makedirs(out_dir, exist_ok=True)

    p = Path(src_motion_dir)
    print("Path: %s"%src_motion_dir)
    for filename in p.iterdir():
        print("Checking out file %s"%filename)
        if filename.suffix == '.bvh':
            load_motion = load_motion_bvh
        elif filename.suffix == '.txt':
            print("Loading 36m file")
            load_motion = load_motion_h36m            
        else:
            continue
            
#        ground_height = 5.5 -1.8 + 0.4 #0#85 
        ground_height = 5.5 -1.8 + 0.2 #0#85 
        ground_height *= 0.01
        src_skeleton, src_motion = load_motion(filename, src_skeleton_type)
        src_scale = 1.0
        if auto_scale:
            min_p, max_p = src_skeleton.get_bounding_box()
            src_height = (max_p[1] - min_p[1])
            src_scale = dest_height / src_height
        
        joint_map = generate_joint_map(src_skeleton.skeleton_model, dest_skeleton.skeleton_model)
        retargeting = Retargeting(src_skeleton, dest_skeleton, joint_map,src_scale, additional_rotation_map=None, place_on_ground=place_on_ground, ground_height=ground_height)        
        
        new_frames = retargeting.run(src_motion.frames, frame_range=None)
        frame_data = convert_quaternion_to_euler_frames(dest_skeleton, new_frames)            
        outfilename = out_dir + os.sep+filename.stem + ".bvh"
        print("write", outfilename, auto_scale, place_on_ground)
        write_euler_frames_to_bvh_file(outfilename, dest_skeleton, frame_data, src_motion.frame_time)

   

test_settings = {
    'src_motion_dir' : "data/bvh/custom_skeleton_walk_example.bvh",
    'src_skel_type' : "mh_cmu",

    'dest_skel' : "data/bvh/mh_cmu_skeleton_pose.bvh",
    'dest_skel_type' : "custom",
    'out_dir' : "out.bvh",

    'h36_path' : '.',
    #'h36_file' : "./data/target/walking_1.txt",
    'h36_file' : "./data/h36m/walking_2.txt",
    'h36_type' : "mh_cmu"
    }


def runmain():
    srmd = test_settings["src_motion_dir"]
    srst = test_settings["src_skel_type"]
    ds = test_settings["dest_skel"]
    dst = test_settings["dest_skel_type"]
    od = test_settings["out_dir"]

    h36path = test_settings['h36_path']
    h36file = test_settings['h36_file']
    h36type = test_settings['h36_type']

    bvhskel, bvhmv = load_motion_bvh(srmd, srst)
    h36skel, hmv = load_motion_h36m(h36file, dst)

    return bvhskel, h36skel, bvhmv, hmv
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run retargeting.')

    parser.add_argument('--ordering', type = str, default = 'XYZ')
    parser.add_argument('dest_skeleton', nargs='?', help='BVH filename')
    parser.add_argument('dest_skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('src_motion_dir', nargs='?', help='src BVH directory')
    parser.add_argument('src_skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('out_dir', nargs='?', help='output BVH directory')
    parser.add_argument('--auto_scale', default=False, dest='auto_scale', action='store_true')
    parser.add_argument('--place_on_ground', default=False, dest='place_on_ground', action='store_true')
    
    args = parser.parse_args()
    if args.src_motion_dir is not None and args.dest_skeleton is not None and args.out_dir is not None:
        print(args.auto_scale)
        print(args.place_on_ground)

        default_channels = [channel_order[c] for c in args.ordering.upper()]
        
        skel = main(args.src_motion_dir, args.src_skeleton_type, args.dest_skeleton, args.dest_skeleton_type, args.out_dir, bool(args.auto_scale), bool(args.place_on_ground))
