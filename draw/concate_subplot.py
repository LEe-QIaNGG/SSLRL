import os
from PIL import Image
import matplotlib.pyplot as plt
source_dir='draw/result/score/'
Type='DA'
# 获取文件夹中所有png图片
image_files = [f for f in os.listdir(source_dir) if f.endswith('.png') and f.startswith(Type)]

# 读取所有图片
images = [Image.open(os.path.join(source_dir, img)) for img in image_files]

# 获取单张图片的宽度和高度
width, height = images[0].size

# 创建新图片,宽度是所有图片宽度之和,高度与单张图片相同
total_width = width * len(images)
result = Image.new('RGB', (total_width, height))

# 横向拼接图片
for i, img in enumerate(images):
    result.paste(img, (i * width, 0))

# 保存结果
result.save(os.path.join(source_dir, f'{Type}_combined_plot.png'))
print(f"图片已保存为 {Type}_combined_plot.png")