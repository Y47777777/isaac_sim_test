import numpy as np
import open3d as o3d

# 读取.npy文件
point_cloud_np = np.load('/home/visionnav/_out_sdrec/pointcloud_0154.npy')
point_cloud_color = np.load('/home/visionnav/_out_sdrec/pointcloud_rgb_0154.npy')

# 分割坐标和颜色信息
points = point_cloud_np[:, :3]  # 点的坐标
colors = point_cloud_color[:, :3]  # 点的颜色

print(points)
# 创建Open3D点云对象
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors/255.0)  

# 保存为.pcd文件
o3d.io.write_point_cloud('/home/visionnav/_out_sdrec/pointcloud_rgb_0154.pcd', point_cloud_o3d)
