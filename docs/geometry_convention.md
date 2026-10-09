# S3 数据集坐标约定

已通过 S3S 第 257 帧的 3D Box 投影验证：

- 相机坐标：X 前、Y 右、Z 上
- translation：3D Box 底面中心
- size：[length, width, height]
- rotation：绕 Z 轴的航向角
- extrinsics_matrix_4x4：标注坐标系 -> 相机坐标系

P_cam = T_json @ P_world
P_world = inverse(T_json) @ P_cam

注意：
- 标注的绝对 Z 坐标不等于离地高度。
- 投影到 BEV 时需要明确地面参考坐标。