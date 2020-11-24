import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
from glob import glob
from Common import point_operation, data_utils as d_utils
import os
warnings.filterwarnings('ignore')
from torchvision import transforms


def load_data(dir,partition="train"):

    all_data = []
    all_label = []

    for h5_name in glob(os.path.join(dir, 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        normal = f['normal'][:].astype('float32')
        f.close()
        data = np.concatenate([data,normal],axis=-1)
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

point_transform = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
    ]
)

class ModelNetDataLoader(Dataset):
    def __init__(self, opts,partition='train'):
        self.opts = opts
        self.data, self.label = load_data(opts.data_dir,partition=partition)
        self.num_points = opts.num_points
        self.partition = partition

        self.dim = 6 if self.opts.use_normal else 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pc = self.data[index][:self.num_points,:self.dim].copy()
        label = self.label[index]
        # np.random.shuffle(pc)
        # if self.partition == 'train':
        #     #np.random.shuffle(pc)
        #     pc = point_transform(pc)
        #     return pc, label.astype(np.int32)

        if self.opts.augment and  self.partition == 'train':
            pc = point_operation.rotate_point_cloud_and_gt(pc)
            pc = point_operation.jitter_perturbation_point_cloud(pc)
            if self.opts.is_dg:
                pc,_ = point_operation.random_scale_point_cloud_and_gt(pc)
                pc = point_operation.rotate_perturbation_point_cloud(pc)
                pc = point_operation.shift_point_cloud_and_gt(pc)
        return pc.astype(np.float32), label.astype(np.int32)
