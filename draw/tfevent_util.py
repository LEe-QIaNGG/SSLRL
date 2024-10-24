import tensorflow as tf
import os
from glob import glob

# 源目录
source_dirs = [
    "log/Seaquest-ram-v4/dqn/sslrl/",
    "log/Seaquest-ram-v4/dqn/baseline/"
]

# 目标目录的基础路径
base_target_dir = "log/Seaquest-ram-v4/dqn/best_reward/"

# 查找所有的 events 文件
event_files = []
for source_dir in source_dirs:
    event_files.extend(glob(os.path.join(source_dir, "**/events.out.tfevents.*"), recursive=True))

# 遍历所有 events 文件
for event_file in event_files:
    # 计算相对路径
    rel_path = os.path.relpath(event_file, start=os.path.commonprefix(source_dirs))
    # 构建目标路径
    target_path = os.path.join(base_target_dir, os.path.dirname(rel_path))
    
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)
    
    # 为每个源文件创建一个新的 FileWriter
    writer = tf.summary.create_file_writer(target_path)
    
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag == "info/best_reward":
                # 使用 writer.as_default() 上下文管理器
                with writer.as_default():
                    tf.summary.scalar("info/best_reward", value.simple_value, step=event.step)
    
    # 刷新并关闭当前 writer
    writer.flush()
    writer.close()

print("提取完成。新的 events 文件已保存在", base_target_dir)
