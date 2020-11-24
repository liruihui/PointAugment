# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:26 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : point_operation.py

import numpy as np

def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)

def shuffle_point_cloud_and_gt(batch_data,batch_gt=None):
    B,N,C = batch_data.shape

    idx = np.arange(N)
    np.random.shuffle(idx)
    batch_data = batch_data[:,idx,:]
    if batch_gt is not None:
        np.random.shuffle(idx)
        batch_gt = batch_gt[:,idx,:]
    return batch_data,batch_gt

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx]

def rotate_point_cloud_by_angle_batch(batch_data, rotation_angle):
  """ Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data

def rotate_point_cloud_and_gt(pc,gt=None,y_rotated=True):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    if y_rotated:
        rotation_matrix = Ry
    else:
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    pc = np.dot(pc, rotation_matrix)
    if gt is not None:
        gt = np.dot(gt, rotation_matrix)
        return pc, gt

    return pc

def jitter_perturbation_point_cloud(pc, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += pc
    return jittered_data

def shift_point_cloud_and_gt(pc, gt = None, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, (3))
    pc = pc + shifts

    if gt is not None:
        gt = gt + shifts
        return pc, gt

    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def random_scale_point_cloud_and_gt(pc, gt = None, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    N, C = pc.shape
    scale = np.random.uniform(scale_low, scale_high, 1)
    pc = pc * scale

    if gt is not None:
        gt = gt * scale
        return pc, gt, scale

    return pc,scale


def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    N, C = pc.shape

    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    pc = np.dot(pc, R)

    return pc

def rotate_2D_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    N, C = pc.shape

    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    R = np.array([
                   [np.cos(angles[0]),-np.sin(angles[0])],
                   [np.sin(angles[0]),np.cos(angles[0])]])
    pc = np.dot(pc, R)

    return pc

def scale_2D_point_cloud(pc, scale_low=0.95, scale_high=1.05):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    N, C = pc.shape
    scale = np.random.uniform(scale_low, scale_high, 1)
    pc = pc * scale

    return pc

def guass_noise_point_cloud(batch_data, sigma=0.005, mu=0.00):
    """ Add guassian noise in per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    batch_data += np.random.normal(mu, sigma, batch_data.shape)
    return batch_data

def rotate_point_cloud_by_angle(self, data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    :param batch_data: Nx3 array, original batch of point clouds
    :param rotation_angle: range of rotation
    :return:  Nx3 array, rotated batch of point clouds
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data


