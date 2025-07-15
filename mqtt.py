import paho.mqtt.client as mqtt
import json
import time

# MQTT 服务器配置
MQTT_BROKER = "me90f19a.ala.cn-hangzhou.emqxsl.cn"
MQTT_PORT = 8883
MQTT_ACC_TOPIC = "sensors/acceleration"
MQTT_GESTURE_TOPIC = "gesture"
MQTT_USERNAME = "Esp8266"
MQTT_PASSWORD = "Esp8266"

# 存储数据的列表
acceleration_data = []
gesture_data = []


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

            acceleration_data.append({
                "timestamp": timestamp,
                "accX": acc_x,
                "accY": acc_y,
                "accZ": acc_z
            })

        elif msg.topic == MQTT_GESTURE_TOPIC:
            # 处理姿势数据（八个身体角度）
            angles = {
                "angle1": data.get("angle1", 0),
                "angle2": data.get("angle2", 0),
                "angle3": data.get("angle3", 0),
                "angle4": data.get("angle4", 0),
                "angle5": data.get("angle5", 0),
                "angle6": data.get("angle6", 0),
                "angle7": data.get("angle7", 0),
                "angle8": data.get("angle8", 0)
            }

            print(f"Received Gesture - Angles: {angles}")

            gesture_data.append({
                "timestamp": timestamp,
                **angles
            })

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