import numpy as np
import open3d as o3
import os
filename = '/home/zhipengz/data/kitti360_pc/s00/pc/0000000001.bin'
pc = np.fromfile(os.path.join('', filename), dtype=np.float32) # roc edit 64->32

# if(len(pc) == 0):
#     print(filename)
#     return np.array([])
# roc edit: random downsample to 4096
'''if (pc.shape[0] % 4096 != 0):
    print("Error in pointcloud shape")
    return np.array([])'''
pc = np.reshape(pc, [-1, 4])
print('len(pc)')
print(pc.shape)
n = np.random.choice(len(pc), 4096, replace=False)
pc = pc[n]

'''if(pc.shape[0] != 4096*4):
    print("Error in pointcloud shape")
    return np.array([])

pc = np.reshape(pc,(pc.shape[0]//4, 4))'''
pc=pc[:, :3]
local_intensity=[]
pcd_nosam = o3.geometry.PointCloud()
pcd_nosam.points = o3.utility.Vector3dVector(pc)
pcd_nosam.colors = o3.utility.Vector3dVector(local_intensity)
o3.io.write_point_cloud('./00.pcd', pcd_nosam)