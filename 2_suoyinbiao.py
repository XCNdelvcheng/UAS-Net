# """
# 建立影像数据与标注文件之间的索引表
# """
# import os
# import glob  # 用于批量查找文件
# import pandas as pd  # 用于创建数据表
#
# # 存储每一行数据（patient_id、CT路径、分割路径）
# rows = []
#
# # 1. 遍历所有NIfTI文件（CT数据）
# for ct_path in glob.glob(r"G:\Nifti\*.nii*"):
#     # 2. 提取患者ID（从CT文件名中解析）
#     # 例如：CT文件名为"001_chenchune.nii.gz"，提取"001"作为patient_id
#     ct_filename = os.path.basename(ct_path)  # 得到文件名（不含路径）
#     patient_id = ct_filename.split('_')[0]  # 按"_"分割，取第一个元素（如"001"）
#
#     # 3. 查找对应的分割文件（.nii.gz）
#     # 搜索分割目录中，文件名包含patient_id的文件（如"001_segmentation.nii.gz"）
#     seg_files = glob.glob(fr"G:\data_0\Segmentation\*{patient_id}*.nii*")
#
#     # 4. 处理分割文件（存在则取路径，不存在则记为None）
#     seg_path = seg_files[0] if seg_files else None  # 若找到多个，取第一个（默认你的数据唯一对应）
#
#     # 5. 保存当前患者的信息
#     rows.append({
#         "patient_id": patient_id,  # 患者编号（如"001"）
#         "ct_path": ct_path,  # CT文件的完整路径
#         "seg_path": seg_path  # 分割文件的完整路径（可选，可能为None）
#     })
#
# # 6. 转换为DataFrame并保存为CSV
# df = pd.DataFrame(rows)
# df.to_csv("dataset_index1.csv", index=False)  # 保存到当前脚本所在目录


"""
建立影像数据与标注文件之间的索引表（按patient_id排序）
"""
import os
import glob
import pandas as pd

rows = []

# 1. 遍历所有NIfTI文件（CT数据）
for ct_path in glob.glob(r"D:\z_up\data\Nifti\*.nii*"):
    # 2. 提取患者ID（从CT文件名中解析）
    ct_filename = os.path.basename(ct_path)
    patient_id = ct_filename.split('_')[0]  # 取"001"部分

    # 3. 查找对应的分割文件
    seg_files = glob.glob(fr"D:\z_up\data\data_0\Segmentation\*{patient_id}*.nii*")
    seg_path = seg_files[0] if seg_files else None

    rows.append({
        "patient_id": patient_id,
        "ct_path": ct_path,
        "seg_path": seg_path
    })

# 4. 转换为DataFrame
df = pd.DataFrame(rows)

# 5. 按 patient_id 排序（考虑ID是字符串形式的“001”、“002”）
# 若 patient_id 为纯数字字符串，需先转为整数再排序
df['patient_id_num'] = df['patient_id'].astype(str).str.extract(r'(\d+)').astype(int)
df = df.sort_values(by='patient_id_num').drop(columns='patient_id_num')

# 6. 保存为CSV
df.to_csv("dataset_index1.csv", index=False, encoding='utf-8-sig')

print("✅ 已生成 dataset_index1.csv，并按 patient_id 从小到大排序。")
