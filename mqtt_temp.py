import paho.mqtt.client as mqtt
import os
import csv
import time
import re

# MQTT 服务器配置
MQTT_BROKER = "me90f19a.ala.cn-hangzhou.emqxsl.cn"
MQTT_PORT = 8883
MQTT_TOPIC = "fake"  # 修改为你的实际主题
MQTT_USERNAME = "Esp8266"
MQTT_PASSWORD = "Esp8266"

# 文件路径
EVAL_FILE = os.path.join("fla", "shot_evaluation.txt")
CSV_FILE = os.path.join("fla", "static", "data", "text2.csv")
HISTORY_CSV_FILE = os.path.join("fla", "static", "data", "text.csv")  # 历史数据文件

# 确保文件和目录存在
os.makedirs(os.path.dirname(EVAL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# 初始化当前数据CSV（text2.csv）
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time(ms)", "ax", "ay", "az"])

# 初始化历史数据CSV（text.csv），仅在文件不存在时创建表头
if not os.path.exists(HISTORY_CSV_FILE):
    with open(HISTORY_CSV_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time(ms)", "ax", "ay", "az", "record_time"])  # 包含记录时间字段

# 用于暂存接收到的多行数据
message_buffer = []

def get_last_timestamp(csv_file):
    """获取CSV文件最后一行的time(ms)值，如果文件为空则返回0.0"""
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        return 0.0
    
    try:
        with open(csv_file, 'r') as f:
            # 使用csv.reader读取最后一行
            reader = csv.reader(f)
            last_row = None
            for row in reader:
                last_row = row
            
            # 跳过表头行
            if last_row and last_row[0] != "time(ms)":
                return float(last_row[0])
            return 0.0
    except Exception as e:
        print(f"读取最后时间戳出错: {str(e)}")
        return 0.0


# 回调函数：连接成功时触发
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code: " + str(rc))
    client.subscribe(MQTT_TOPIC, 0)


# 回调函数：接收到消息时触发
def on_message(client, userdata, msg):
    global message_buffer
    try:
        # 接收消息并按行分割
        payload = msg.payload.decode('utf-8').strip()
        # 清洗消息：移除 { } 和 \r
        payload = re.sub(r'[{}]', '', payload).replace('\r', '')
        lines = [line.strip() for line in payload.split('\n') if line.strip()]

        print(f"Received lines: {lines}")  # 调试：打印接收到的行

        # 添加到缓冲区
        message_buffer.extend(lines)

        # 当缓冲区达到或超过9行时处理
        if len(message_buffer) >= 9:
            # 取出前9行
            current_message = message_buffer[:9]
            message_buffer = message_buffer[9:]  # 保留未处理的行
            print(f"Processing message: {current_message}")  # 调试：打印处理的消息

            # 第一行是完整的评价数据
            evaluation = current_message[0].strip()
            # 验证第一行不是加速度数据（不含逗号）
            if ',' in evaluation:
                print(f"Error: First line '{evaluation}' looks like acceleration data, skipping")
                return

            # 保存评价到 shot_evaluation.txt
            with open(EVAL_FILE, "a", encoding="utf-8") as f:
                f.write(evaluation + "\n")

            # 获取历史数据最后一个时间戳
            last_time = get_last_timestamp(HISTORY_CSV_FILE)
            # 计算新数据的起始时间戳
            start_time = last_time + 20.0
            print(f"Last timestamp: {last_time}, Next start timestamp: {start_time}")

            # 处理后八行的加速度数据（第2到第9行）
            # 处理后八行的加速度数据（第2到第9行）
            accel_data = []
            current_time = 0.0  # text2.csv 时间戳从 0 开始
            last_time = get_last_timestamp(HISTORY_CSV_FILE)  # 获取 text.csv 的最后时间戳
            history_time = last_time + 20.0  # 用于 text.csv 的时间戳

            for line in current_message[1:9]:
                try:
                    # 确保每行有4个逗号分隔的数值
                    if line.count(',') != 3:
                        print(f"Error: Invalid acceleration data format: {line}")
                        continue
                    # 只提取 ax, ay, az，忽略原始时间戳
                    _, ax, ay, az = map(float, line.split(','))
                    # 使用新的时间戳
                    accel_data.append([current_time, ax, ay, az])
                    # 下一行时间戳增加 20ms
                    current_time += 20.0
                except ValueError as e:
                    print(f"Error parsing acceleration data: {line}, {str(e)}")
                    continue

            # 1. 更新当前数据 CSV（text2.csv） - 覆盖模式，只保留最新数据
            with open(CSV_FILE, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time(ms)", "ax", "ay", "az"])  # 写入表头
                for row in accel_data:
                    writer.writerow(row)

            # 2. 添加到历史数据 CSV（text.csv） - 追加模式，保留所有数据
            record_time = time.time()
            with open(HISTORY_CSV_FILE, "a", newline='') as f:
                writer = csv.writer(f)
                for row in accel_data:
                    # 使用 history_time 保持 text.csv 的时间戳连续性
                    writer.writerow([history_time] + row[1:] + [record_time])
                    history_time += 20.0  # 历史时间戳递增

            print(f"Saved evaluation: {evaluation}")
            print(f"Saved {len(accel_data)} rows to current CSV and history CSV")

    except Exception as e:
        print(f"Error processing message: {str(e)}")


# 初始化 MQTT 客户端
client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

# 启用 TLS
client.tls_set()

# 连接服务器
try:
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
except Exception as e:
    print(f"Failed to connect to MQTT Broker: {str(e)}")
    exit(1)

# 启动循环
client.loop_forever()
