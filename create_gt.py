import os.path as osp
import pickle
import os
import open3d as o3d
import numpy as np
import glob
# from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import shutil
import random
def load_poses(path):
    poses = list()
    with open(path, "r") as f:
        for line in f.readlines():
            values = line.split(" ")
            if len(values) < 9:
                continue
            r = R.from_quat([float(values[5]), float(values[6]), float(values[7]), float(values[8])])
            T = np.zeros((4, 4))
            T[0:3, 0:3] = r.as_matrix()
            T[0:4, 3:4] = np.asarray([[float(values[2])], [float(values[3])], [float(values[4])], [1]])
            poses.append(np.mat(T))
    return poses
def crop_pcd(pcd, max_len):
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_max = 30
    vol.axis_min = -2
    bounding_ploy = np.array([
        [max_len, 0, 0],
        [0, max_len, 0],
        [-max_len, 0, 0],
        [0, -max_len, 0]
    ], dtype=np.float64)
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_ploy)
    return vol.crop_point_cloud(pcd)

def main():
    file_names = glob.glob(osp.join("/media/w/wangqian/F/data/2023-05-26_12_28_02/keyframes/", '*.pcd'))
    file_names.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    poses_path = osp.join("/media/w/wangqian/F/data/2023-05-26_12_28_02/loop", 'update_pose.txt')
    poses = load_poses(poses_path)
    datas_train = list()
    datas_test = list()
    datas_val = list()
    last_file_name = None
    last_frame_id = None

    frame_count = len(file_names)
    train_cutoff = int(frame_count * 3 / 4)
    test_cutoff = train_cutoff + int(frame_count / 8)

    # for file_name in tqdm(file_names):

    for file_index, file_name in enumerate(file_names):
    # for i in range(0, len(file_names), frame_interval):
        # file_name      = file_names[i]
        frame_id       = int(file_name.split('/')[-1][:-4])
        # frame_interval = random.randint(1, 10)
        # frame_interval = 1
        
        # if file_index % frame_interval == 0:
            # new_file_name = osp.join('/mnt/share_disk/wq/hibag_20230223_2_lio/downsampled', '{:02d}'.format(0), '{:06d}'.format(frame_id) + '.npy')
        new_file_name = osp.join('/media/w/wangqian/F/data/2023-05-26_12_28_02/downsampled', '04_frame1' if file_index < train_cutoff else '05_frame1' if file_index < test_cutoff else '06_frame1', '{:06d}'.format(frame_id) + '.npy')
        pcd = o3d.io.read_point_cloud(file_name)
        max_len = 70.0  # 设置裁剪的最大长度
        pcd_cropped = crop_pcd(pcd, max_len)
        # pcd = pcd.voxel_down_sample(0.5)
        pcd_cropped = pcd_cropped.voxel_down_sample(0.5)
        points = np.array(pcd_cropped.points).astype(np.float32)
        np.save(new_file_name, points)
        if last_file_name is not None:
            transform = np.asarray(poses[last_frame_id].I * poses[frame_id])
            if file_index  < train_cutoff:
                datas_train.append({
                    "seq_id": 2,
                    "frame0": last_frame_id,
                    "frame1": frame_id,
                    "transform": transform,
                    "pcd0": last_file_name,
                    "pcd1": new_file_name,
                })
                transform_file_path = "/media/w/wangqian/F/data/2023-05-26_12_28_02/transform/train_1/{:06d}-{:06d}.npy".format(last_frame_id, frame_id)
                np.save(transform_file_path, transform)


            elif file_index  < train_cutoff:
                datas_test.append({
                    "seq_id": 2,
                    "frame0": last_frame_id,
                    "frame1": frame_id,
                    "transform": transform,
                    "pcd0": last_file_name,
                    "pcd1": new_file_name,
                })
                transform_file_path = "/media/w/wangqian/F/data/2023-05-26_12_28_02/transform/test_1/{:06d}-{:06d}.npy".format(last_frame_id, frame_id)
                np.save(transform_file_path, transform)


            else:
                datas_val.append({
                    "seq_id": 2,
                    "frame0": last_frame_id,
                    "frame1": frame_id,
                    "transform": transform,
                    "pcd0": last_file_name,
                    "pcd1": new_file_name,
                })
                transform_file_path = "/media/w/wangqian/F/data/2023-05-26_12_28_02/transform/val_1/{:06d}-{:06d}.npy".format(last_frame_id, frame_id)
                np.save(transform_file_path, transform)
        last_file_name = new_file_name
        last_frame_id = frame_id

        with open("/media/w/wangqian/F/data/2023-05-26_12_28_02/metadata/train_1.pkl", 'wb') as f:
            pickle.dump(datas_train, f)
        with open("/media/w/wangqian/F/data/2023-05-26_12_28_02/metadata/test_1.pkl", 'wb') as f:
            pickle.dump(datas_test, f)
        with open("/media/w/wangqian/F/data/2023-05-26_12_28_02/metadata/val_1.pkl", 'wb') as f:
            pickle.dump(datas_val, f)


if __name__ == '__main__':
    main()