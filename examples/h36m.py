import numpy as np
from csv import reader
import torch
import math

from anim_utils.animation_data.bvh import write_euler_frames_to_bvh_file


euler_params = {
    'XYZ' : { 'sign' : - 1,
              'v' : [1, 2, 0],
              'order' : [0, 1, 2]
             },
    
    'XZY' : { 'sign' : 1,
              'v' : [2, 1, 0],
              'order' : [0, 2, 1]
              },
    
    'YXZ' : { 'sign' : 1,
              'v' : [0, 2, 1],
              'order' : [1, 0, 2]
             },
    'YZX' : { 'sign' : -1,
              'v' : [2, 0, 1],
              'order' : [1, 2, 0]
             },
    'ZXY' : { 'sign' : -1,
              'v' : [0, 1, 2],
              'order' : [2, 0, 1]
             },
    'ZYX' : { 'sign' : 1,
              'v' : [1, 0, 2],
              'order' : [2, 1, 0]
             }
}

class RotationEuler:
    def __init__(self, order):
        self.order = order
        self.params = euler_params[order]
    
        self.sign = self.params['sign']
        #self.v0,self.v1,self.v2 = self.params['v']
        
    def rot_to_euler(self, R, degrees = False, reorder = False):
        v1, v2, v0 = self.params['v']
        i0, i1, i2 = self.params['order']

        n = R.data.shape[0]
        eul = np.zeros([n, 3])

        idx_spec1 = (R[:, v0, v2] == 1).nonzero()[0].reshape(-1).tolist()
        idx_spec2 = (R[:, v0, v2] == -1).nonzero()[0].reshape(-1).tolist()


        if (len(idx_spec1) > 0):
            R_spec1 = R[idx_spec1, :, :]
            eul_spec1 = np.zeros([len(idx_spec1), 3])
            eul_spec1[:, i2] = 0
            eul_spec1[:, i1] = self.sign * np.pi / 2

            delta = np.arctan2(R_spec1[:, v0, v1], R_spec1[:, v0, v2])

            eul_spec1[:, i0] = delta
            eul[idx_spec1, :] = eul_spec1

        elif(len(idx_spec2) > 0):
            R_spec2 = R[idx_spec2, :, :]
            eul_spec2 = np.zeros([len(idx_spec2), 3])
            eul_spec2[:, i2] = 0
            eul_spec2[:, i1] = - self.sign * np.pi / 2
            eul_spec2[:, i0] = delta
            eul[idx_spec2, :] = eul_spec2

        idx_remain = np.arange(0, n)
        idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()

        if (len(idx_remain) > 0):
            R_remain = R[idx_remain, :, :]
            eul_remain = np.zeros([len(idx_remain), 3])
            eul_remain[:, i1] = - np.arcsin(self.sign * R_remain[:, v0, v2])
            # eul_remain[:, i0] = np.arctan2(self.sign * R_remain[:, v1, v2] / np.cos(eul_remain[:, i1]),
            #                                R_remain[:, v2, v2] / np.cos(eul_remain[:, i1]))
            
            # eul_remain[:, i2] = np.arctan2(self.sign * R_remain[:, v0, v1] / np.cos(eul_remain[:, i1]),
            #                                R_remain[:, v0, v0] / np.cos(eul_remain[:, i1]))
            
            # eul[idx_remain, :] = eul_remain


            # eul_asin = np.arcsin(self.sign * R_remain[:, v0, v2])

            # ww = np.where(eul_asin < 0, math.pi + eul_asin, eul_asin)
            # eul_remain[:, i1] = - eul_asin

            eul_0num = - self.sign * R_remain[:, v1, v2] / np.cos(eul_remain[:, i1])
            eul_0den = R_remain[:, v2, v2] / np.cos(eul_remain[:, i1])
            eul_2num = - self.sign * R_remain[:, v0, v1] / np.cos(eul_remain[:, i1])
            eul_2den = R_remain[:, v0, v0] / np.cos(eul_remain[:, i1])        

            # print("0num: ", eul_0num)
            # print("0den: ", eul_0den)
            # print("2num: ", eul_2num)
            # print("2den: ", eul_2den)        

            #signchange = np.where(eul_0num < 0, 1, -1)

            eul_remain[:, i0] = np.arctan2(eul_0num, eul_0den)
            eul_remain[:, i2] = np.arctan2(eul_2num, eul_2den)

            eul[idx_remain, :] = eul_remain
            
        if (reorder):
            eul = eul[:, self.params['order']]
            
        if degrees:
            return eul * 180 / math.pi
        else:            
            return eul
    
    def rot_to_euler_xyz(self, rotmat):

        n = R.data.shape[0]
        eul = np.zeros([n, 3])
    
        idx_spec1 = (R[:, 0, 2] == 1).nonzero()[0].reshape(-1).tolist()
        idx_spec2 = (R[:, 0, 2] == -1).nonzero()[0].reshape(-1).tolist()
    
        if (len(idx_spec1) > 0):
            R_spec1 = R[idx_spec1, :, :]
            eul_spec1 = np.zeros([len(idx_spec1), 3])
            eul_spec1[:, 2] = 0
            eul_spec1[:, 1] = -np.pi / 2
            delta = np.arctan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
            eul_spec1[:, 0] = delta
            eul[idx_spec1, :] = eul_spec1
    
        elif(len(idx_spec2) > 0):
            R_spec2 = R[idx_spec2, :, :]
            eul_spec2 = np.zeros([len(idx_spec2), 3])
            eul_spec2[:, 2] = 0
            eul_spec2[:, 1] = -np.pi / 2
            delta = np.arctan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
            eul_spec2[:, 0] = delta
            eul[idx_spec2, :] = eul_spec2
    
        idx_remain = np.arange(0, n)
        idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    
        if (len(idx_remain) > 0):
            R_remain = R[idx_remain, :, :]
            eul_remain = np.zeros([len(idx_remain), 3])
        
            eul_remain[:, 1] = -np.arcsin(R_remain[:, 0, 2])
            eul_remain[:, 0] = np.arctan2(R_remain[:, 1, 2] / np.cos(eul_remain[:, 1]),
                                          R_remain[:, 2, 2] / np.cos(eul_remain[:, 1]))
            
            eul_remain[:, 2] = np.arctan2(R_remain[:, 0, 1] / np.cos(eul_remain[:, 1]),
                                          R_remain[:, 0, 0] / np.cos(eul_remain[:, 1]))
            
            eul[idx_remain, :] = eul_remain
    
        #return 180.0 / math.pi * eul
        return  eul
        


def _some_variables():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
    We define some variables that are useful to run the kinematic tree
    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lengths
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30])
    
    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    
    offset = offset.reshape(-1, 3)
    offset[:, 2] = -offset[:, 2]
    offset[:, 0] = -offset[:, 0]    
    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    bone_names = ['Hip',
                  'RHip',
                  'RKnee',
                  'RFoot',
                  'RFootTip',
                  'RToe',
                  'LHip',
                  'LKnee',
                  'LFoot',
                  'LFootTip',
                  'LToe',
                  'LowerSpine',
                  'Spine',
                  'Thorax',
                  'Nose',
                  'Head',
                  'LClavicle',
                  'LShoulder',
                  'LElbow',
                  'LWrist',
                  'LHand',
                  'LThumb',
                  'LFinger1',
                  'LFinger2',
                  'RClavicle',
                  'RShoulder',
                  'RElbow',
                  'RWrist',
                  'RHand',
                  'RThumb',
                  'RFinger1',
                  'RFinger2'
                  ]


    return parent, offset, rotInd, expmapInd, bone_names


# Model


# cos_map: (from Erik Herrmann)
# This is a dictionary containing a map from joint names to x and y vectors defining a local coordinate system of the respective joint. It is used by the retargeting code to retarget between two different skeletons without a t-pose by finding a global correcting rotation after transforming the vectors of the source and target joints into the global coordinate system. The global corection can then be brought back into the local coordinate system to be exported again.




def generate_skeleton_model():
    model = dict()

    model['name'] = "human36m"
    model['model'] = {
        'flip_x_axis' : False,
        'cos_map' : {}
        # 'foot_joints' : [],
        # 'joints' : {},
        # 'joint_constraints' : {},
        # 'foot_correction' : {'x' : 0, 'y' : 0 },
        # 'ik_chains' : {},
        # 'aligning_root_node' : "",
        # 'heel_offset' : [],
        # 'relative_head_dir' : []
        # 'free_joints_map': {}
        }

    parents, _, _, _, bone_names = _some_variables().parents()

    
    return model

    

def fkl(rotmat, parent, offset, rotInd, expmapInd):
    """
    convert joint angles to joint locations
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = np.tile(np.expand_dims(offset, 0), [n, 1, 1])
    
    #R = rotmat.view(n, j_n, 3, 3)
    R = rotmat.reshape([n, j_n, 3, 3])
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = np.matmul(R[:, i, :, :], R[:, parent[i], :, :])
            p3d[:, i, :] = np.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]

    return p3d


def rotmat2xyz(rotmat):
    """
    convert expmaps to joint locations
    :param rotmat: N*32*3*3
    :return: N*32*3
    """

    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd, bonenames = _some_variables()
    xyz = fkl(rotmat, parent, offset, rotInd, expmapInd)
    return xyz


def rotmat2euler_aborted(R):
    # Algorithm from https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    R_31 = R[:, 2, 0]
    
    theta1 = -np.arcsin(R[:, 2, 0])

    psi1 = np.arctan2(R[:, 2, 1] / np.cos(theta1), R[:, 2, 2] / np.cos(theta1))
    phi1 = np.arctan2(R[:, 1, 0] / np.cos(theta1), R[:, 0, 0] / np.cos(theta1))
    
    # theta2 = math.pi - theta1    
    # psi2 = np.arctan2(R[:, 2, 1] / np.cos(theta2), R[:, 2, 2] / np.cos(theta2))    
    # phi2 = np.arctan2(R[:, 1, 0] / np.cos(theta2), R[:, 0, 0] / np.cos(theta2))    

    #theta01 = 0.5 * math.pi * np.ones_like(theta1)


    #theta0neg1= -0.5 * math.pi * np.ones_like(theta1)
    psi01    =   0 + np.arctan2(  R[:, 0, 1],   R[:, 0, 2])
    psi0neg1 = - 0 + np.arctan2(- R[:, 0, 1], - R[:, 0, 2])

    
    phi = np.where(R_31 * R_31 == 1, 0, phi1)

    theta = np.where( R_31 * R_31 == -1,
                     R[:, 2, 0] * math.pi,
                     theta1)

    psi = np.where(R[:, 2, 0] == 1,
                   psi01,
                   np.where(R[:, 2, 0] == -1,
                            psi0neg1,
                            psi01))
    

    
    return np.array([theta, psi, phi]).transpose()

def rotmat2euler(R):

    # XYZ format. Try to get these in 'ZXY'?
    n = R.data.shape[0]
    eul = np.zeros([n, 3])

    idx_spec1 = (R[:, 0, 2] == 1).nonzero()[0].reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero()[0].reshape(-1).tolist()

    if (len(idx_spec1) > 0):
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = np.zeros([len(idx_spec1), 3])
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = np.arctan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    elif(len(idx_spec2) > 0):
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = np.zeros([len(idx_spec2), 3])
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = np.arctan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2, :] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()

    if (len(idx_remain) > 0):
        R_remain = R[idx_remain, :, :]
        eul_remain = np.zeros([len(idx_remain), 3])

        eul_remain[:, 1] = -np.arcsin(R_remain[:, 0, 2])
        eul_remain[:, 0] = np.arctan2(R_remain[:, 1, 2] / np.cos(eul_remain[:, 1]),
                                     R_remain[:, 2, 2] / np.cos(eul_remain[:, 1]))
        
        eul_remain[:, 2] = np.arctan2(R_remain[:, 0, 1] / np.cos(eul_remain[:, 1]),
                                     R_remain[:, 0, 0] / np.cos(eul_remain[:, 1]))
        
        eul[idx_remain, :] = eul_remain
    return -1.0 * eul


def rotmat2euler_old( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.transpose([0, 2, 1])
  r = np.zeros_like(rotdiff[:, 0])

  r[:, 0] = -rotdiff[:, 1, 2]
  r[:, 1] = rotdiff[:, 0, 2]
  r[:, 2] = -rotdiff[:, 0, 1]

  r_norm = np.linalg.norm(r, axis = 1)
  sintheta = 0.5 * r_norm

  rnb = np.expand_dims(r_norm, 1)
  r0 = np.divide(r, rnb.repeat(3, axis = 1) + 0.00000001)

  t1 = R[:, 0, 0]
  t2 = R[:, 1, 1]
  t3 = R[:, 2, 2]

  costheta = (t1 + t2 + t3 - 1) / 2

  theta = np.arctan2(sintheta, costheta)

  q = np.zeros([R.shape[0], 4])
  q[:, 0] = np.cos(theta / 2)

  qnb = np.expand_dims(np.sin(theta / 2), 1).repeat(3, axis = 1)
  
  q[:, 1:] = r0 * qnb
  return q


def rotmat2quat_old(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    
    Args
    r: n x 1x3 exponential map
    Returns
    R: n x 3x3 rotation matrix
    """
    theta = np.linalg.norm(r, axis = 1)
    r0 = np.divide(r.T, theta + np.finfo(np.float32).eps).T
    r0x = np.zeros([r.shape[0], 3, 3])    
    r0x[:,0, 1] = -r0[:, 2]
    r0x[:,0, 2] = r0[:, 1]
    r0x[:,1, 2] = -r0[:, 0]

    r0x[:,1, 0] = r0[:, 2]
    r0x[:,2, 0] = -r0[:, 1]
    r0x[:,2, 1] = r0[:, 0]
    Ri = np.zeros([r0x.shape[0], 3, 3])
    Ri[:] = np.eye(3)
    

    Rsin = np.sin(theta).repeat(9).reshape(-1, 3, 3) * r0x

    Rcos = (1 - np.cos(theta)).repeat(9).reshape(-1, 3, 3) * np.matmul(r0x, r0x)
    
    R = Ri + Rsin + Rcos
    return R

def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r


def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.
    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = torch.from_numpy(offset).float().to(rotmat.device).unsqueeze(0).repeat(n, 1, 1).clone()
    R = rotmat.view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    return p3d

def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above
    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = torch.zeros(n, 3).float().to(R.device)
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()

    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = torch.zeros(len(idx_spec1), 3).float().to(R.device)
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = torch.zeros(len(idx_spec2), 3).float().to(R.device)
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = torch.zeros(len(idx_remain), 3).float().to(R.device)
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))

        eul[idx_remain, :] = eul_remain
        
    return eul

def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = torch.zeros(R.shape[0], 4).float().to(R.device)
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q

def expmap2quat(exp):
    theta = np.expand_dims(np.linalg.norm(exp, 1), 1)
    v = np.divide(exp, theta.repeat(3, axis = 1) + 0.0000001)
    sinhalf = np.sin(theta / 2)
    coshalf = np.cos(theta / 2)
    q1 = v * sinhalf.repeat(3, axis = 1)
    q = np.concatenate([coshalf, q1], axis = 1)
    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    Ri = torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device)
    Rsin = torch.mul(torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1)
    Rcos =  torch.mul((1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    
    R = Ri + Rsin + Rcos
    return R

def rotmat2xyz_torch(rotmat):
    """
    convert expmaps to joint locations
    :param rotmat: N*32*3*3
    :return: N*32*3
    """
    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd, bone_names = _some_variables()
    xyz = fkl_torch(rotmat, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2xyz(expmap):
    rotmat = np.expand_dims(expmap2rotmat(expmap[1:, :]), 0)
    return rotmat2xyz(rotmat).squeeze(0)


class Human36MReader:
    """ Class for reading the Human3.6m Dataset
    
    Parameters
    ----------
    * filename : string\tPath to the motion data that is initially loaded
    """
    # The skeleton is hardcoded, which is odd for a mocap dataset built from multiple subjects
    # Human 3.6m motion data is organized as a traditional csv file
    
    def __init__(self, filename = ""):

        if (filename != ""):
            with open(filename, "r") as fp:
                csvreader = reader(fp)
                expmap = [[float(c) for c in r] for r in csvreader]
                
                self.expmap = np.array(expmap).reshape([len(expmap), -1, 3])
                
        self.parentjoints, self.offsets, _, _, self.bone_names = _some_variables()

        self.build_tree()

    def nonendnodes(self):
        return [i for i, b in enumerate(self.bone_names) if len(self.tree[b]) > 0]
        
    def framecount(self):
        return self.expmap.shape[0]

    def get_quaternion_frames(self, prune_list = None):
        if (prune_list):
            v = self.nonendnodes()
            return np.array([rotmat2quat(expmap2rotmat(self.expmap[i, prune_list, :]) )for i in range(self.framecount())])
        else:
            return np.array([rotmat2quat(expmap2rotmat(self.expmap[i, :, :]) )for i in range(self.framecount())])        

    # def get_euler_frames(self, prune_list = None):
    #     if (prune_list):
    #         ef = np.array([rotmat2euler(expmap2rotmat(self.expmap[i, prune_list, :]) )for i in range(self.framecount())])            
    #     else:
    #         ef = np.array([rotmat2euler(expmap2rotmat(self.expmap[i, :, :]) )for i in range(self.framecount())])
    #     return (180/math.pi) * ef

    def preprocess_frames(self):
        # Flip the axis so that the y component is always positive
        q = np.where(self.expmap[:, :, 1] <  0, -1, -1)
        qq = np.stack([q, q, q], axis = 2)
        return qq * self.expmap

    
    def get_euler_frames(self, prune_list = None, order = 'XYZ', reorder = False):

        rotter = RotationEuler(order)

        pexpmap = self.preprocess_frames()
        
        if (prune_list):
            ef = np.array([rotter.rot_to_euler(expmap2rotmat(pexpmap[i, prune_list, :]), reorder = reorder) for i in range(self.framecount())])            
        else:
            ef = np.array([rotter.rot_to_euler(expmap2rotmat(pexpmap[i, :, :]), reorder = reorder)for i in range(self.framecount())])


        ef[:, 1:, :]  = (180/math.pi) * ef[:, 1:, :]
        ef[:, 0, :] =  self.expmap[:, 0, :]
        return ef


    def get_euler_frames_torch(self, prune_list = None):
        if (prune_list):
            
            tlist = [rotmat2euler_torch(expmap2rotmat_torch(torch.tensor(self.expmap[i, prune_list, :]))) for i in range(self.framecount())]
            return np.array([t.numpy() for t in tlist])
        else:    
            tlist = [rotmat2euler_torch(expmap2rotmat_torch(torch.tensor(self.expmap[i, :, :]))) for i in range(self.framecount())]
            return np.array([t.numpy() for t in tlist])
    
    def get_keypoints(self, prune_list = None):
        if (prune_list):
            return np.array([expmap2xyz(self.expmap[i, prune_list, :]) for i in range(self.framecount())])
        else:
            return np.array([expmap2xyz(self.expmap[i, :, :]) for i in range(self.framecount())])

    def build_tree(self):
        self.tree = {}
        for c, p in enumerate(self.parentjoints):
            if (p >= 0):
                if (self.bone_names[p] in self.tree):
                    self.tree[self.bone_names[p]].append(self.bone_names[c])
                else:
                    self.tree[self.bone_names[p]] = [self.bone_names[c]]

                if not (c in self.bone_names):
                    self.tree[self.bone_names[c]] = []
            else:
                self.tree['ROOT'] = [self.bone_names[c]]
                self.root_bone = self.bone_names[c]


    def dump_to_bvh(self, filename = None):
        # Maybe the easy route to a conversion pipeline is batch convert to bvh and then retarget from there using EHerr's other tools
        pass

                
if __name__ == '__main__':
    # a = np.array([-0.087784,0.1520913,1.8149083,
    #               -1.277117,-0.5005686,1.7185123,
    #               -1.7023442,-0.8528787,1.3199044])

    a = np.array([-0.0883684,0.1523683,1.8164481,
                  -0.0895846,0.1513751,1.8158307,
                  -0.0932234,0.1496553,1.816841,
                  -0.0901167,0.1497043,1.8159411,
                  -0.087784,0.1520913,1.8149083,
                  -1.277117,-0.5005686,1.7185123,
                  -1.6958656,-0.8454831,1.3300213,
                  -1.6982317,-0.8518177,1.324792,
                  -1.7023442,-0.8528787,1.3199044,
                  -1.7274348,-0.8428802,1.3137,
                  -1.702204,-0.852534,1.3141199,
                  -1.7128727,-0.8592606,1.298571,
                  -1.7131711,-0.8551665,1.3039329,
                  -1.7105469,-0.8436885,1.3281237,
                  -1.7238495,-0.8390244,1.3215078,
                  -1.724807,-0.825124,1.3320475,
                  -1.6956774,-0.8398946,1.3300215,
                  -1.698384,-0.8407102,1.3255851,
                  -0.6854224,-0.1182413,1.8573706,
                  -0.5703838,-0.0597103,1.8667015,
                  -0.4851359,-0.0172013,1.8671561,
                  -0.1378789,0.1729176,1.8238779,
                  -0.1773616,0.1512383,1.8366871,
                  -0.2100935,0.1325715,1.8459357])
                  
    np.set_printoptions(suppress = True)
    a = np.reshape(a, [-1, 3])
    print("A shape is ", a.shape)
    ar = expmap2rotmat(a)
    
    print(a)
    print("-- Rotmat --")
    arp = np.reshape(ar, [ar.shape[0], -1])
    #print(arp)
    for i in range(arp.shape[0]):
        print(",".join([str(k) for k in arp[i, :]]))

    #for o in euler_params.keys():
    for o in ['XYZ']:
        print("--%s--"%o)        
        rotter = RotationEuler(o)
        ae = rotter.rot_to_euler(ar, degrees = True)
        print(ae)


