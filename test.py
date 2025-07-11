import mediapipe as mp

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # 只追踪一只手
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_hand_center(frame):
    """返回手部中心坐标（x, y），未检测到则返回None"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # 取第一只手的关键点
        hand_landmarks = results.multi_hand_landmarks[0]
        # 提取掌心坐标（可选用手腕或中指根部，这里用掌心平均坐标）
        palm_landmarks = [4, 8, 12, 16, 20]  # 指尖关键点索引
        x_list = [hand_landmarks.landmark[i].x for i in palm_landmarks]
        y_list = [hand_landmarks.landmark[i].y for i in palm_landmarks]
        center_x = sum(x_list) / len(x_list)  # 归一化x（0~1）
        center_y = sum(y_list) / len(y_list)  # 归一化y（0~1）

        # 转换为像素坐标（假设画面尺寸640x480）
        frame_height, frame_width = frame.shape[:2]
        center_x_pixel = int(center_x * frame_width)
        center_y_pixel = int(center_y * frame_height)

        # 绘制手部关键点和中心
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.circle(frame, (center_x_pixel, center_y_pixel), 10, (0, 255, 0), -1)
        return frame, (center_x_pixel, center_y_pixel)

    return frame, None  # 未检测到手