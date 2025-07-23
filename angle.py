import ast
import math
import time
import os
from DTW import process_hit

class SlidingWindow:
    def __init__(self, size=8):
        self.size = size
        self.window = []

    def add_data(self, data):
        if len(self.window) < self.size:
            self.window.append(data)
        else:
            self.window.pop(0)
            self.window.append(data)

    def get_window(self):
        return self.window

def calculate_magnitude(data):
    if len(data) == 3:
        return math.sqrt(data[0] ** 2 + data[1] ** 2 + data[2] ** 2)
    return 0

def process_hit_moment(pose_window, accel_window, timestamp):
    print("处理击球瞬间数据：")
    print("时间戳:", timestamp)
    print("Pose窗口数据:", pose_window)
    print("加速度窗口数据:", accel_window)
    with open("hit_moments.txt", "a", encoding="utf-8") as output_file:
        output_data = {"timestamp": timestamp, "pose": pose_window, "accel": accel_window}
        output_file.write(str(output_data) + "\n")
    process_hit(pose_window, accel_window, timestamp)

def monitor_files(pose_file_path, accel_file_path):
    pose_window = SlidingWindow(size=8)
    accel_window = SlidingWindow(size=8)
    pose_offset = 0
    accel_offset = 0

    while True:
        try:
            with open(pose_file_path, "r", encoding="utf-8") as pose_file:
                pose_file.seek(pose_offset)
                new_pose_lines = pose_file.readlines()
                pose_offset = pose_file.tell()

                with open(accel_file_path, "r", encoding="utf-8") as accel_file:
                    accel_file.seek(accel_offset)
                    new_accel_lines = accel_file.readlines()
                    accel_offset = accel_file.tell()

                    for pose_line, accel_line in zip(new_pose_lines, new_accel_lines):
                        try:
                            pose_data = ast.literal_eval(pose_line.strip())
                            accel_data = ast.literal_eval(accel_line.strip())
                            timestamp = time.time()

                            pose_window.add_data(pose_data)
                            accel_window.add_data(accel_data)

                            current_pose_window = pose_window.get_window()
                            current_accel_window = accel_window.get_window()

                            print("Pose窗口内容:", current_pose_window)
                            print("加速度窗口内容:", current_accel_window)

                            if len(current_pose_window) == 8 and len(current_accel_window) == 8:
                                magnitudes = []
                                for i in range(2, 5):
                                    mag = calculate_magnitude(current_accel_window[i])
                                    magnitudes.append(mag)
                                    print(f"第{i + 1}个加速度数据的合加速度: {mag:.4f}")

                                if magnitudes and magnitudes[1] == max(magnitudes):
                                    print("检测到击球瞬间！(第4个数据合加速度为峰值)")
                                    process_hit_moment(current_pose_window, current_accel_window, timestamp)
                                else:
                                    print("未检测到击球瞬间。")

                        except Exception as e:
                            print(f"解析数据失败: {e}")

            time.sleep(0.1)
        except Exception as e:
            print(f"监控文件错误: {e}")
            time.sleep(1)

if __name__ == "__main__":
    pose_file_path = "pose_data.txt"
    accel_file_path = "tennis_acceleration_data.txt"
    print("开始监控pose和加速度数据文件...")
    monitor_files(pose_file_path, accel_file_path)