import os
import json
from pathlib import Path
import numpy as np
import argparse

from h36m import Human36MReader, _some_variables

from anim_utils.animation_data import BVHReader, MotionVector, SkeletonBuilder   

from anim_utils.animation_data.skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE

from anim_utils.animation_data.constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER

from anim_utils.animation_data.bvh import write_euler_frames_to_bvh_file, convert_quaternion_to_euler_frames

from anim_utils.animation_data import Skeleton



channel_order = {'X' : 'Xrotation',
                 'Y' : 'Yrotation',
                 'Z' : 'Zrotation' }
#default_channels = ['Zrotation', 'Xrotation', 'Yrotation']
#root_channels = ['Xposition', 'Yposition', 'Zposition'] + default_channels
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

def load_motion_from_h36mreader(mv, h36mreader, filter_joints = True, animated_joints = True, fps = 60.0, order = 'XYZ'):
    # H36m is in Exponential map, so needs conversion with both formats
    if (mv.rotation_type == ROTATION_TYPE_QUATERNION):
        quat_frames = h36mreader.get_quaternion_frames()
        mv.frames = quat_frames.reshape([quat_frames.shape[0], -1])
        
    elif (mv.rotation_type == ROTATION_TYPE_EULER):
        euler_frames = h36mreader.get_euler_frames(order = order)
        mv.frames = euler_frames.reshape([euler_frames.shape[0], -1])

    mv.n_frames = 0
    mv._prev_n_frames = 0
    mv.frame_time = 1.0 / fps


# Remember to convert ExpMap to Euler Angles
def construct_hierarchy_from_h36m(skeleton, h36m, node_name, level, default_channels):

    if (node_name == skeleton.root):
        #print("Default chans: ", default_channels)
        root_channels = ['Xposition', 'Yposition', 'Zposition'] + default_channels
        node = SkeletonRootNode(h36m.tree[node_name][0], root_channels, None, level)
    elif node_name in h36m.tree:
        node = SkeletonJointNode(node_name, default_channels, None, level)
    else:
        node = SkeletonEndSiteNode(node_name, default_channels, None, level)

    nodeidx = h36m.bone_names.index(node_name)
    node.index = nodeidx
    node.offset = list(0.1 * h36m.offsets[nodeidx, :])

    # Initial Unit Quaternion. From BVH, it looks like it may come from a reference frame
    node.rotation = np.array([1.0, 0.0, 0.0, 0.0])
        
    if node_name in skeleton.animated_joints:
        is_fixed = False
        #quaternion_frame_index
        
    joint_index = -1

    skeleton.nodes[node_name] = node
    if (node_name in h36m.tree):
        for c in h36m.tree[node_name]:
            new_node = construct_hierarchy_from_h36m(skeleton, h36m, c, level + 1, default_channels)
            new_node.parent = node
            node.children.append(new_node)
    return node


def load_skeleton_from_h36mreader(h36mreader, default_channels, skeleton_type = None):
    skeleton = Skeleton()
    # Not sure if it's wise to have 'ROOT' as the name of the root bone
    skeleton.root = h36mreader.tree['ROOT'][0]
    #print("Setting root to %s"%skeleton.root)
    nodes = construct_hierarchy_from_h36m(skeleton, h36mreader, h36mreader.tree['ROOT'][0], 0, default_channels) 
    create_euler_frame_indices(skeleton)
    SkeletonBuilder.set_meta_info(skeleton)
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton



def get_motion_selection(nodelist, node, body_list, level):
    # Mimics the order of the BVH hierarchy in order to pick out the frames used by the bvh
    joint_output = []
    test_joint_output = []

    if (len(nodelist[node].children) > 0):
        joint_output.append(body_list.index(node))
        test_joint_output.append(node)

    for child in nodelist[node].children:
        j, tj = get_motion_selection(nodelist, child.node_name, body_list, level + 1)
        joint_output.extend(j)
        test_joint_output.extend(tj)
    
    return joint_output, test_joint_output

def load_motion_h36m(h36file, default_channels, skeleton_type = None, fps = 60.0, order = 'XYZ', reorder = False):
    h36mreader = Human36MReader(h36file)

    
    mv = MotionVector()
    h36skel = load_skeleton_from_h36mreader(h36mreader, default_channels, skeleton_type)

    joints_list, _ = get_motion_selection(h36skel.nodes, h36skel.root, h36mreader.bone_names, 0)

    load_motion_from_h36mreader(mv, h36mreader, fps = fps, order = order)
    frame_data = np.array(h36mreader.get_euler_frames(prune_list = joints_list, order = order, reorder = reorder))
    return h36skel, mv, frame_data



def main(infile, outfile, default_channels, order = 'XYZ',  fps = 60.0, noroot = False, norootpos = False, claviclefix = False, reorder = False):
    p = Path(infile)
   
    skel, motion, frame_data = load_motion_h36m(infile, default_channels, "h36m", order = order, reorder = reorder)

    out_root_pos = np.zeros([frame_data.shape[0], 3])

    out_rot_data = frame_data.reshape([frame_data.shape[0], -1])
    out_rot_data = np.concatenate([out_rot_data, out_root_pos], axis = 1)

    if (noroot):
        out_rot_data[:, 3:6] = 0.00

    if (norootpos):
        out_rot_data[:, :3] = 0.00

    if (claviclefix):
        bn = _some_variables()[4]
        lclidx = bn.index("LClavicle")
        rclidx = bn.index("RClavicle")

        out_rot_data[:, 3 * (lclidx + 1) + 2] = out_rot_data[:, 3 * (lclidx + 1) + 2] - 90
        out_rot_data[:, 3 * (rclidx + 1) + 2] = out_rot_data[:, 3 * (rclidx + 1) + 2] + 90        
        
    write_euler_frames_to_bvh_file(outfile, skel, out_rot_data , 1.0 / fps)

def dump_keypoints(infile, outfile):
    h36reader = Human36MReader(infile)
    kp = h36reader.get_keypoints()
    np.savez(outfile, poses = h36reader.expmap, keypoints = kp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run retargeting.')
    parser.add_argument('--ordering', type = str, default = 'XYZ')    
    parser.add_argument("--kp", type = str, help = "Dump keypoints to a file")
    parser.add_argument("--fps", type = float, help = "Override fps", default = 50.0)
    parser.add_argument("--norootpos", action = "store_true", help = "No root motion")
    parser.add_argument("--testreorder", action = "store_true", help = "Reorder root pos")
    parser.add_argument("--claviclefix", action = "store_true", help = "Alter clavicle rotations")
    
    parser.add_argument('--norootori',action="store_true", help = "No root rotation")
    parser.add_argument('infile', nargs='?', help='H36M filename')


    #parser.add_argument('skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('output_file', type = str, help = "output file name")

    args = parser.parse_args()

    if (args.kp):
        dump_keypoints(args.infile, args.kp)

    default_channels = [channel_order[c] for c in args.ordering.upper()]

    main(args.infile, args.output_file, default_channels, order = args.ordering.upper(), fps = args.fps, noroot = args.norootori, norootpos = args.norootpos, claviclefix = args.claviclefix, reorder = args.testreorder)

    
