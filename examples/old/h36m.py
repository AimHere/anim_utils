import numpy as np
from csv import reader
import torch


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
    print("Len of parents is %d"%len(parent))
    
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
                  'Neck/Nose',
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

    R = rotmat.view(n, j_n, 3, 3)
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
    

def rotmat2euler(R):
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
    return eul

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

    def framecount(self):
        return self.expmap.shape[0]

    def get_quaternion_frames(self):
        return np.array([rotmat2quat(expmap2rotmat(self.expmap[i, :, :]) )for i in range(self.framecount())])        

    def get_euler_frames(self):
        return np.array([rotmat2euler(expmap2rotmat(self.expmap[i, :, :]) )for i in range(self.framecount())])

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
                
if __name__ == '__main__':
    a = np.random.random([32, 3]).astype(np.float32)
    at = torch.from_numpy(a)
    atr = expmap2rotmat_torch(at)
    ar = expmap2rotmat(a)

    teu = rotmat2euler_torch(atr)

    print(teu)

    eu = rotmat2euler(ar)
    print("--")
    print(eu)
