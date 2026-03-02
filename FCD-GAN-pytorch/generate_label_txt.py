"""
生成 FCD-GAN 项目所需的 label.txt 文件
支持 WHU 和 LEVIR-MCI 数据集
"""

import os
import numpy as np
from PIL import Image


def generate_label_txt(label_dir, output_txt, image_ext='.png'):
    """
    遍历 label 目录，根据标签图是否包含变化区域生成 label.txt
    
    参数:
        label_dir: label 目录路径
        output_txt: 输出的 label.txt 路径
        image_ext: 图像扩展名
    """
    if not os.path.exists(label_dir):
        print(f"错误：目录不存在 - {label_dir}")
        return
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir) 
                   if f.endswith(image_ext) or f.endswith('.tif')]
    label_files.sort()  # 排序保证一致性
    
    if len(label_files) == 0:
        print(f"警告：目录中没有找到图像文件 - {label_dir}")
        return
    
    print(f"开始处理：{label_dir}")
    print(f"找到 {len(label_files)} 个标签文件")
    
    changed_count = 0
    unchanged_count = 0
    
    with open(output_txt, 'w') as f:
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            
            try:
                # 读取标签图
                label_img = Image.open(label_path)
                label_np = np.array(label_img)
                
                # 检查是否有变化区域 (像素值 > 0)
                has_change = np.sum(label_np > 0) > 0
                
                # 变化标志：1=有变化，0=无变化
                change_flag = 1 if has_change else 0
                
                if has_change:
                    changed_count += 1
                else:
                    unchanged_count += 1
                
                # 写入 label.txt
                # 格式：文件名，类别 1，类别 2，变化标签 (0/1)
                f.write(f"{label_file},0,0,{change_flag}\n")
                
            except Exception as e:
                print(f"处理失败 {label_file}: {e}")
                continue
    
    print(f"完成！输出文件：{output_txt}")
    print(f"  有变化样本：{changed_count}")
    print(f"  无变化样本：{unchanged_count}")
    print(f"  总计：{changed_count + unchanged_count}\n")


def main():
    # 数据集配置列表
    datasets = [
        {
            'name': 'WHU_CDC',
            'label_dir': r'D:\project\CD\Dataset\whu_CDC_dataset_converted\label',
            'output_txt': r'D:\project\CD\wscd\FCD-GAN-pytorch\label_whu.txt'
        },
        {
            'name': 'LEVIR-MCI',
            'label_dir': r'D:\project\CD\Dataset\LEVIR-MCI-dataset_converted\label',
            'output_txt': r'D:\project\CD\wscd\FCD-GAN-pytorch\label_levir.txt'
        }
    ]
    
    print("=" * 60)
    print("FCD-GAN label.txt 生成脚本")
    print("=" * 60 + "\n")
    
    for dataset in datasets:
        print(f"[{dataset['name']}]")
        generate_label_txt(
            label_dir=dataset['label_dir'],
            output_txt=dataset['output_txt']
        )
    
    print("=" * 60)
    print("所有数据集处理完成！")
    print("=" * 60)
    print("\n生成的文件可直接用于 FCD-GAN 项目，修改 Demo_WSSS.py 中的路径：")
    print("  LabelDir = r'数据集根目录'  # label.txt 所在目录")
    print("  ImgDirX = r'数据集根目录/before'")
    print("  ImgDirY = r'数据集根目录/after'")
    print("  RefDir = r'数据集根目录/label'")


if __name__ == '__main__':
    main()
