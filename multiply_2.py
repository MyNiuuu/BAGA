import os
import cv2
import numpy as np

def process_images(folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 确保是图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 读取图片
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                
                if image is None:
                    print(f"无法读取图片: {file_path}")
                    continue
                
                # 将图片像素值乘以2并限制到255
                processed_image = np.clip(image * 2, 0, 255).astype(np.uint8)
                
                # 保存并覆盖处理后的图片
                cv2.imwrite(file_path, processed_image)
                # print(f"处理并保存图片: {file_path}")
            except Exception as e:
                print(f"处理图片时出错: {file_path}, 错误: {e}")


scenes = ['000', '001', '002', '003', '004', '005', '006', '007']
cameras = ['19305338', '19305337', '19061154', '19061151', '19305311', '19305316', '19305317', '19305328', '19305326', '19061156', '19308875', '19305322']
for scene in scenes:
    for camera in cameras:
        # 替换成你的图片文件夹路径
        folder_path = f"/home/yinqiang/nmy/BAGA/data/BSHuman/{scene}/images/{camera}"
        process_images(folder_path)
