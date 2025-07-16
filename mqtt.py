import paho.mqtt.client as mqtt
import json
import time
import os

# MQTT 服务器配置
MQTT_BROKER = "me90f19a.ala.cn-hangzhou.emqxsl.cn"
MQTT_PORT = 8883
MQTT_ACC_TOPIC = "sensors/acceleration"
MQTT_GESTURE_TOPIC = "gesture"
MQTT_USERNAME = "Esp8266"
MQTT_PASSWORD = "Esp8266"

# 文件路径
ACCEL_FILE = "tennis_acceleration_data.txt"
POSE_FILE = "pose_data.txt"

# 确保文件存在
if not os.path.exists(ACCEL_FILE):
    open(ACCEL_FILE, "a").close()
if not os.path.exists(POSE_FILE):
    open(POSE_FILE, "a").close()

# 回调函数：连接成功时触发
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code: " + str(rc))
    # 订阅加速度和姿势主题
    client.subscribe([(MQTT_ACC_TOPIC, 0), (MQTT_GESTURE_TOPIC, 0)])

# 回调函数：接收到消息时触发
def on_message(client, userdata, msg):
    try:
        # 解析 JSON 数据
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        timestamp = data.get("timestamp", time.time())

        if msg.topic == MQTT_ACC_TOPIC:
            # 处理加速度数据
            acc_x = data.get("accX", 0)
            acc_y = data.get("accY", 0)
            acc_z = data.get("accZ", 0)

            print(f"Received Acceleration - AccX: {acc_x}, AccY: {acc_y}, AccZ: {acc_z}")

            # 保存到文件
            with open(ACCEL_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{acc_x}, {acc_y}, {acc_z}]\n")

        elif msg.topic == MQTT_GESTURE_TOPIC:
            # 处理姿势数据（八个身体角度）
            angles = [
                data.get("angle1", 0),
                data.get("angle2", 0),
                data.get("angle3", 0),
                data.get("angle4", 0),
                data.get("angle5", 0),
                data.get("angle6", 0),
                data.get("angle7", 0),
                data.get("angle8", 0)
            ]

            print(f"Received Gesture - Angles: {angles}")

            # 保存到文件
            with open(POSE_FILE, "a", encoding="utf-8") as f:
                f.write(f"{angles}\n")

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON data")
    except Exception as e:
        print(f"Error: {str(e)}")

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