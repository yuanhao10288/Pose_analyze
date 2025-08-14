import cv2
import mediapipe as mp
import numpy as np
import paho.mqtt.client as mqtt
import time
import json
from flask import Flask, Response
import threading
import requests

# MQTT配置
MQTT_BROKER = "me90f19a.ala.cn-hangzhou.emqxsl.cn"  # 替换为你的服务器地址
MQTT_PORT = 8883  # 默认 SSL 端口（如果是 1883，改用非 SSL）
MQTT_TOPIC = "gesture"
MQTT_USERNAME = "Esp8266"  # 替换为你的用户名
MQTT_PASSWORD = "Esp8266"  # 替换为你的密码

# Flask服务配置
app = Flask(__name__)
latest_frame = None  # 存储最新帧的全局变量

#树莓派信息传输配置

# 树莓派IP地址和端口号，需根据实际情况修改
# url = "http://192.168.157.206:5000/receive_coordinate"

def calculate_angle(a, b, c):
    '''
    计算角度
    :param a:
    :param b:
    :param c:
    :return:
    '''

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_dist(a, b):
    '''
    计算欧式距离
    :param a:
    :param b:
    :return:
    '''
    #该处考虑改为绝对距离
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)#默认参数(矩阵2范数，不保留矩阵二维特性)
    return dist


# MQTT连接回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT连接成功")
    else:
        print(f"MQTT连接失败，错误码: {rc}")

# MQTT发布回调函数
def on_publish(client, userdata, mid):
    print(f"数据发布成功，消息ID: {mid}")

# 创建MQTT客户端
client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# 设置回调函数
client.on_connect = on_connect
client.on_publish = on_publish

# 配置TLS连接
client.tls_set()  # 如果使用SSL连接，需要调用此方法

# 连接MQTT服务器
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()  # 启动MQTT循环线程
except Exception as e:
    print(f"MQTT连接错误: {e}")
    exit()

@app.route('/opencv-stream')
def stream():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                # 将OpenCV帧转换为JPG格式
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                # 按HTTP流协议返回
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 启动Flask服务线程
def start_flask_server():
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

# 创建并启动Flask服务线程
flask_thread = threading.Thread(target=start_flask_server, daemon=True)
flask_thread.start()


# 替换为树莓派的实际IP和端口
stream_url = "http://192.168.88.205:8081"

# 创建VideoCapture对象
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 打开mediapipe
mp_pose = mp.solutions.pose
pose=mp_pose.Pose()
# static_image_mode = False,
# model_complexity = 0,  # 0=最低复杂度，适合嵌入式设备
# smooth_landmarks = True,
# min_detection_confidence = 0.5,
# min_tracking_confidence = 0.5
mp_drawing = mp.solutions.drawing_utils

publish_interval = 0.12  # 发布间隔(秒)
last_publish_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，尝试重新连接...")
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        continue
    results=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        # 获取相应关键点的坐标
        # 左肩11
        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        # 左手肘13
        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        # 左手腕15
        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # 左胯23
        lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        # 左膝盖25
        lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        # 左脚踝27
        lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        # 右肩膀12
        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        # 右手肘14
        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        # 右手腕16
        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        # 右胯24
        rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        # 右膝盖26
        rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        # 右脚踝28
        rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        # 鼻子
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        # 计算角度和距离
        langle = calculate_angle(lshoulder, lelbow, lwrist)  # 左胳膊角度
        rangle = calculate_angle(rshoulder, relbow, rwrist)  # 右胳膊角度
        lsangle = calculate_angle(lhip, lshoulder, lelbow)  # 左臂离身体角度
        rsangle = calculate_angle(rhip, rshoulder, relbow)  # 右臂离身体角度
        lhangle = calculate_angle(lshoulder, lhip, lknee)  # 左肩膀-胯-膝盖角度
        rhangle = calculate_angle(rshoulder, rhip, rknee)  # 右肩膀-胯-膝盖角度
        lkangle = calculate_angle(lankle, lknee, lhip)  # 左腿角度
        rkangle = calculate_angle(rankle, rknee, rhip)  # 右腿角度
        newdis = calculate_dist(nose, rwrist)
        # newangle = calculate_angle(relbow, lshoulder, nose)  # 右肘部到鼻子到左肩部的角度
        # ankdist = calculate_dist(lankle, rankle)  # 左右脚踝距离
        # rwdist = calculate_dist(rhip, rwrist)  # 右胯到右手腕距离
        # lwdist = calculate_dist(lhip, lwrist)  # 左胯到左手腕距离

        # test.append(
        #     [langle, rangle, lsangle, rsangle, lhangle, rhangle, lkangle, rkangle])
        test = [langle, rangle, lsangle, rsangle, lhangle, rhangle, lkangle, rkangle]
        # print(test)

        # coordinates = rwrist  # 假设这是要发送的坐标
        # data = {"x": coordinates[0], "y": coordinates[1]}
        # response = requests.post(url, json=data)
        # print(response.text)
        # print("右手腕坐标：",rwrist)
        # MQTT:构建要发布的数据
        gesture_data = {
            "timestamp": time.time(),
            "angles": {
                "left_arm": langle,
                "right_arm": rangle,
                "left_arm_body": lsangle,
                "right_arm_body": rsangle,
                "left_hip": lhangle,
                "right_hip": rhangle,
                "left_knee": lkangle,
                "right_knee": rkangle
            }

        }

        # 打印数据
        print(json.dumps(gesture_data, indent=2))

        # 控制发布频率
        current_time = time.time()
        if current_time - last_publish_time >= publish_interval:
            try:
                # 将数据转换为JSON格式并发布到MQTT
                client.publish(MQTT_TOPIC, json.dumps(gesture_data))
                last_publish_time = current_time
            except Exception as e:
                print(f"MQTT发布错误: {e}")

    latest_frame = frame
    # 显示帧
    # cv2.imshow('MJPG Stream', frame)

    # 按'q'退出
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

client.loop_stop()  # 停止MQTT循环线程
client.disconnect()  # 断开MQTT连接
# cap.release()
# cv2.destroyAllWindows()