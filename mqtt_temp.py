import paho.mqtt.client as mqtt
import os
import csv
import time
import re

# MQTT 服务器配置
MQTT_BROKER = "me90f19a.ala.cn-hangzhou.emqxsl.cn"
MQTT_PORT = 8883
MQTT_TOPIC = "message"  # 修改为你的实际主题
MQTT_USERNAME = "Esp8266"
MQTT_PASSWORD = "Esp8266"

# 文件路径
EVAL_FILE = os.path.join("fla", "shot_evaluation.txt")
CSV_FILE = os.path.join("fla", "static", "data", "text.csv")

# 确保文件和目录存在
os.makedirs(os.path.dirname(EVAL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time(ms)", "ax", "ay", "az"])

# 用于暂存接收到的多行数据
message_buffer = []


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
            message_buffer = message_buffer[9:]  # 清空已处理的行
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

            # 后八行是加速度数据（第2到第9行）
            accel_data = []
            for line in current_message[1:9]:
                try:
                    # 确保每行有4个逗号分隔的数值
                    if line.count(',') != 3:
                        print(f"Error: Invalid acceleration data format: {line}")
                        continue
                    time_ms, ax, ay, az = map(float, line.split(','))
                    accel_data.append([time_ms, ax, ay, az])
                except ValueError as e:
                    print(f"Error parsing acceleration data: {line}, {str(e)}")
                    continue

            # 清空 text.csv（保留表头），然后写入新数据
            with open(CSV_FILE, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time(ms)", "ax", "ay", "az"])  # 写入表头
                for row in accel_data:
                    writer.writerow(row)

            print(f"Saved evaluation: {evaluation}")
            print(f"Saved {len(accel_data)} rows of acceleration data to CSV")

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