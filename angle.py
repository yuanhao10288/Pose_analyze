import ast
import math


class SlidingWindow:
    def __init__(self, size=8):
        self.size = size
        self.window = []

    def add_data(self, data):
        # 如果窗口未满，直接添加
        if len(self.window) < self.size:
            self.window.append(data)
        else:
            # 窗口已满，移除最早的数据，添加新数据
            self.window.pop(0)
            self.window.append(data)

    def get_window(self):
        # 返回当前窗口内容
        return self.window


def calculate_magnitude(data):
    # 计算合加速度：sqrt(x^2 + y^2 + z^2)
    if len(data) == 3:
        return math.sqrt(data[0] ** 2 + data[1] ** 2 + data[2] ** 2)
    return 0


def process_hit_moment(pose_window, accel_window):
    # 处理击球瞬间的两个窗口数据
    print("处理击球瞬间数据：")
    print("Pose窗口完整数据:", pose_window)
    print("加速度窗口完整数据:", accel_window)
    # 这里可以添加更多的处理逻辑，例如保存到文件或进行分析
    # 为了演示，简单地将数据写入文件
    with open("hit_moments.txt", "a") as output_file:
        output_data = [pose_window, accel_window]
        output_file.write(str(output_data) + "\n")


def read_pose_data(file_path):
    # 初始化滑动窗口
    window = SlidingWindow(size=8)

    try:
        with open(file_path, "r") as file:
            for line in file:
                # 将字符串形式的数组转换为Python列表
                data = ast.literal_eval(line.strip())
                window.add_data(data)

                # 打印当前窗口内容
                print("Pose窗口内容:", window.get_window())

                # 返回当前窗口内容以供同步使用
                yield window.get_window()

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
    except Exception as e:
        print(f"错误: {str(e)}")


def read_acceleration_data(pose_file_path, accel_file_path):
    # 初始化滑动窗口
    accel_window = SlidingWindow(size=8)
    pose_generator = read_pose_data(pose_file_path)

    try:
        with open(accel_file_path, "r") as accel_file:
            for line, pose_window in zip(accel_file, pose_generator):
                # 将字符串形式的数组转换为Python列表
                accel_data = ast.literal_eval(line.strip())
                accel_window.add_data(accel_data)

                current_accel_window = accel_window.get_window()
                print("加速度窗口内容:", current_accel_window)

                # 当加速度窗口满8个数据时，检查击球瞬间
                if len(current_accel_window) == 8 and len(pose_window) == 8:
                    magnitudes = []
                    for i in range(2, 5):  # 第3,4,5个数据（索引2,3,4）
                        mag = calculate_magnitude(current_accel_window[i])
                        magnitudes.append(mag)
                        print(f"第{i + 1}个加速度数据的合加速度: {mag:.4f}")

                    # 判断第4个数据是否为击球瞬间
                    if magnitudes[1] == max(magnitudes):
                        print("检测到击球瞬间！(第4个数据合加速度为峰值)")
                        # 调用处理函数，传入两个窗口的完整数据
                        process_hit_moment(pose_window, current_accel_window)
                    else:
                        print("未检测到击球瞬间。")

    except FileNotFoundError:
        print(f"错误: 文件 {accel_file_path} 或 {pose_file_path} 未找到")
    except Exception as e:
        print(f"错误: {str(e)}")


# 测试代码
if __name__ == "__main__":
    pose_file_path = "pose_data.txt"
    accel_file_path = "tennis_acceleration_data.txt"
    print("处理pose和加速度数据：")
    read_acceleration_data(pose_file_path, accel_file_path)