#python run_retargeting.py data\bvh\mh_cmu_skeleton_pose.bvh mh_cmu data\bvh\custom_skeleton_walk_example.bvh custom out.bvh
#python run_retargeting.py data\target\mh_cmu_skeleton_pose.bvh mh_cmu data\bvh\ custom out.bvh
#python run_retargeting.py data\target\mh_cmu_skeleton_pose.bvh mh_cmu data\h36m\ custom out.bvh
#python test_convert.py data\target\mh_cmu_skeleton_pose.bvh mh_cmu data\h36m\ custom out.bvh
python test_convert.py data\target\mh_cmu_skeleton_pose.bvh mh_cmu data\h36m\ h36m out.bvh
