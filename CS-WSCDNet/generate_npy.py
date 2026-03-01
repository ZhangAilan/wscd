import os
import numpy as np
from PIL import Image

def generate_npy(label_folder, output_path):
    """
    根据像素级标签生成图像级标签 npy 文件
    
    参数:
        label_folder: 标签文件夹路径 (二值掩膜，白色表示变化)
        output_path: 输出 npy 文件路径
    """
    label_dict = {}
    
    for filename in os.listdir(label_folder):
        if filename.endswith(".png"):
            label_path = os.path.join(label_folder, filename)
            label = Image.open(label_path).convert("L")
            label_array = np.array(label)
            
            # 计算白色像素占比
            white_pixel_count = np.sum(label_array > 0)
            total_pixels = label_array.size
            white_pixel_percentage = (white_pixel_count / total_pixels) * 100
            
            # 阈值判断：白色像素 >= 0.30% 则为变化 (1)，否则为无变化 (0)
            label_value = 1 if white_pixel_percentage >= 0.30 else 0

            # 提取图像名 (去除 .png 后缀)
            img_name = filename.replace(".png", "")

            # 尝试转换为整数，如果失败则保留字符串格式
            try:
                dict_key = int(img_name)
            except ValueError:
                dict_key = img_name

            # 生成 2 类 multi-hot 标签：[无变化，有变化]
            label = np.zeros(2, dtype=np.int32)
            label[label_value] = 1
            label_dict[dict_key] = label
    
    np.save(output_path, label_dict)
    print(f"已生成 {len(label_dict)} 个图像标签")
    print(f"保存至：{output_path}")


if __name__ == "__main__":
    # 修改为你的标签文件夹路径
    label_folder = r"D:\project\CD\Dataset\LEVIR-MCI-dataset_converted\label"
    
    # 输出路径
    output_path = r"D:\project\CD\wscd\CS-WSCDNet\dataset\LEVIR\levir.npy"
    
    generate_npy(label_folder, output_path)
