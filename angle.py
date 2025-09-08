import ast
import math
import time
import os
from DTW import process_hit

# 全局时间戳计数器，单位为毫秒
global_time_ms = 0.0
# 上一次写入 shot_evaluation.txt 的时间戳
last_shot_time = 0.0

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
    global global_time_ms, last_shot_time  # 使用全局时间戳计数器和最后击球时间
    print("处理击球瞬间数据：")
    print("时间戳:", timestamp)
    print("Pose窗口数据:", pose_window)
    print("加速度窗口数据:", accel_window)

    # 获取 shot_id（与 shot_evaluation.txt 同步）
    shot_id = 1
    output_file = "fla/shot_evaluation.txt"
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                try:
                    last_id = lines[-1].split(" ", 1)[0]
                    shot_id = int(last_id) + 1 if last_id.isdigit() else 1
                except Exception as e:
                    print(f"警告：无法解析最后一行ID，设置为默认值1。错误：{e}")

    # 写入 hit_moments.txt
    with open("hit_moments.txt", "a", encoding="utf-8") as output_file:
        output_data = {"timestamp": timestamp, "pose": pose_window, "accel": accel_window}
        output_file.write(str(output_data) + "\n")

    # 写入加速度数据到 CSV
    CSV_FILE = os.path.join("fla", "static", "data", "text.csv")
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)  # 确保目录存在
    with open(CSV_FILE, "a", encoding="utf-8") as csv_file:
        # 如果文件为空，写入表头
        if os.path.getsize(CSV_FILE) == 0:
            csv_file.write("time(ms),ax,ay,az\n")
        # 写入八个加速度数据，使用连续时间戳
        for i, accel in enumerate(accel_window):
            time_ms = global_time_ms + i * 20.0  # 每次增加 20ms
            ax, ay, az = accel
            csv_file.write(f"{time_ms:.1f},{ax:.1f},{ay:.1f},{az:.1f}\n")

    # 额外写入击球时刻的加速度数据到 text2.csv
    CSV_FILE2 = os.path.join("fla", "static", "data", "text2.csv")
    os.makedirs(os.path.dirname(CSV_FILE2), exist_ok=True)  # 确保目录存在
    
    # 清空文件内容，只保留表头
    with open(CSV_FILE2, "w", encoding="utf-8") as csv_file2:
        csv_file2.write("time(ms),ax,ay,az\n")
    
    # 追加写入击球数据
    with open(CSV_FILE2, "a", encoding="utf-8") as csv_file2:
        # 写入八个加速度数据，使用连续时间戳
        for i, accel in enumerate(accel_window):
            time_ms = global_time_ms + i * 20.0  # 每次增加 20ms
            ax, ay, az = accel
            csv_file2.write(f"{time_ms:.1f},{ax:.1f},{ay:.1f},{az:.1f}\n")

    # 计算并写入合加速度到 hit_magnitudes.txt
    MAGNITUDE_FILE = os.path.join("fla", "hit_magnitudes.txt")
    os.makedirs(os.path.dirname(MAGNITUDE_FILE), exist_ok=True)  # 确保目录存在
    with open(MAGNITUDE_FILE, "a", encoding="utf-8") as mag_file:
        # 如果文件为空，写入表头
        if os.path.getsize(MAGNITUDE_FILE) == 0:
            mag_file.write("shot_id,timestamp,magnitude1,magnitude2,magnitude3,magnitude4,magnitude5,magnitude6,magnitude7,magnitude8\n")
        # 计算每个加速度数据的合加速度
        magnitudes = [calculate_magnitude(accel) for accel in accel_window]
        # 格式化输出，保留4位小数
        mag_str = ",".join([f"{mag:.4f}" for mag in magnitudes])
        mag_file.write(f"{shot_id},{timestamp},{mag_str}\n")

    # 更新全局时间戳计数器
    global_time_ms += 160.0  # 8 个数据点，每次增加 20ms，共 160ms

    # 检查与上一次击球的时间差，决定是否调用 process_hit
    if timestamp - last_shot_time >= 5:
        print(f"调用 process_hit，更新 shot_evaluation.txt (shot_id: {shot_id})")
        process_hit(pose_window, accel_window, timestamp)
        last_shot_time = timestamp  # 更新最后击球时间
    else:
        print(f"跳过 process_hit，距离上一次击球 {timestamp - last_shot_time:.2f} 秒，未达到5秒间隔")

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

                                # 检查第4个数据点的合加速度是否为峰值且大于10
                                if (magnitudes and magnitudes[1] == max(magnitudes) and magnitudes[1] > 10):
                                    print(f"检测到击球瞬间！(第4个数据合加速度为峰值: {magnitudes[1]:.4f})")
                                    process_hit_moment(current_pose_window, current_accel_window, timestamp)
                                else:
                                    reason = []
                                    if not magnitudes or magnitudes[1] != max(magnitudes):
                                        reason.append("第4个数据合加速度非峰值")
                                    if not magnitudes or magnitudes[1] <= 10:
                                        reason.append(f"第4个数据合加速度({magnitudes[1] if magnitudes else 'N/A'} <= 10)")
                                    print(f"未检测到击球瞬间。原因: {', '.join(reason)}")

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