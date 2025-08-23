import ast
import os
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from numpy.linalg import norm
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib

# 加载正反手分类模型和标准化 “
shot_type_model_path = "svm_model/optimized_tennis_pose_svm.pkl"
shot_type_scaler_path = "svm_model/scaler.pkl"
shot_type_clf = joblib.load(shot_type_model_path)
shot_type_scaler = joblib.load(shot_type_scaler_path)

# 加载OneClassSVM模型和标准化器（正手和反手）
oneclass_forehand_model_path = "svm_model/oneclass_svm_forehand.pkl"
oneclass_backhand_model_path = "svm_model/oneclass_svm_backhand.pkl"
oneclass_forehand_scaler_path = "svm_model/scaler_forehand.pkl"
oneclass_backhand_scaler_path = "svm_model/scaler_backhand.pkl"
oneclass_svm_forehand = joblib.load(oneclass_forehand_model_path)
oneclass_svm_backhand = joblib.load(oneclass_backhand_model_path)
oneclass_scaler_forehand = joblib.load(oneclass_forehand_scaler_path)
oneclass_scaler_backhand = joblib.load(oneclass_backhand_scaler_path)

# 加载训练数据以提取标准均值
data = pd.read_csv('data.csv')
features = ['langle', 'rangle', 'lsangle', 'rsangle', 'lhangle', 'rhangle', 'lkangle', 'rkangle']
data_forehand = data[data['class'] == 0][features].values
data_backhand = data[data['class'] == 1][features].values
std_mean_forehand = np.mean(data_forehand, axis=0)
std_mean_backhand = np.mean(data_backhand, axis=0)
print("正手标准角度均值：", std_mean_forehand)
print("反手标准角度均值：", std_mean_backhand)

# 正手标准角度与加速度
standard_body_angles_forehand = np.array([
  [175.3656341825068, 149.64132931817562, 2.774788212212442, 15.082576361448309, 179.00518082835598, 173.25134983909663, 155.11829720083617, 172.99395526839447],
  [168.06101654169362, 141.57231589261391, 1.5077292395458906, 13.648307764773993, 178.0137267266813, 177.16588445775744, 165.85057664101336, 171.89308386279484],
  [176.82746755519403, 147.99156828469742, 14.643990499359608, 12.982746275808568, 177.40033909980434, 172.66217080832217, 172.48419586282057, 165.29032699701926],
  [173.4881183131736, 157.22403864973953, 16.33722195868832, 13.734914198574268, 177.80294263691812, 166.86327056243476, 171.99333527753834, 173.28338608949934],
  [161.50578895348454, 152.73979464401114, 14.051587599492747, 18.1844606980548, 176.7136205873657, 163.06337218295448, 171.950808634663, 176.94309784472438],
  [156.46306617196072, 153.85326534292298, 7.2313526994123185, 11.364519884119076, 177.79516139867278, 171.17773800359404, 170.802994523598, 177.05542171519386],
  [166.93228997676073, 152.23419995117808, 10.742254355261306, 10.355940555803008, 166.69603760559502, 176.73558496294137, 162.11491380775664, 174.2107679514437],
  [170.88299324152166, 142.53910783397174, 0.2941442398258775, 10.049530338557245, 160.22659211951768, 163.01186927204114, 144.8332541529328, 162.88567551140426]
])
standard_acceleration_forehand = np.array([
    [0.2, 0.1, 0.5],
    [0.3, 0.2, 0.6],
    [0.5, 0.2, 1.0],
    [0.8, 0.3, 2.0],
    [1.2, 0.5, 3.5],
    [0.9, 0.4, 2.0],
    [0.6, 0.2, 1.0],
    [0.3, 0.1, 0.5],
])

# 反手标准角度与加速度
standard_body_angles_backhand = np.array([
  [175.3656341825068, 149.64132931817562, 2.774788212212442, 15.082576361448309, 179.00518082835598, 173.25134983909663, 155.11829720083617, 172.99395526839447],
  [168.06101654169362, 141.57231589261391, 1.5077292395458906, 13.648307764773993, 178.0137267266813, 177.16588445775744, 165.85057664101336, 171.89308386279484],
  [176.82746755519403, 147.99156828469742, 14.643990499359608, 12.982746275808568, 177.40033909980434, 172.66217080832217, 172.48419586282057, 165.29032699701926],
  [173.4881183131736, 157.22403864973953, 16.33722195868832, 13.734914198574268, 177.80294263691812, 166.86327056243476, 171.99333527753834, 173.28338608949934],
  [161.50578895348454, 152.73979464401114, 14.051587599492747, 18.1844606980548, 176.7136205873657, 163.06337218295448, 171.950808634663, 176.94309784472438],
  [156.46306617196072, 153.85326534292298, 7.2313526994123185, 11.364519884119076, 177.79516139867278, 171.17773800359404, 170.802994523598, 177.05542171519386],
  [166.93228997676073, 152.23419995117808, 10.742254355261306, 10.355940555803008, 166.69603760559502, 176.73558496294137, 162.11491380775664, 174.2107679514437],
  [170.88299324152166, 142.53910783397174, 0.2941442398258775, 10.049530338557245, 160.22659211951768, 163.01186927204114, 144.8332541529328, 162.88567551140426]
])
standard_acceleration_backhand = np.array([
    [0.25, 0.1, 0.4],
    [0.35, 0.15, 0.7],
    [0.6, 0.2, 1.2],
    [0.9, 0.3, 2.3],
    [1.4, 0.55, 3.8],
    [1.0, 0.45, 2.2],
    [0.7, 0.25, 1.1],
    [0.35, 0.12, 0.6],
])

# 标签名
angle_labels = [
    "左胳膊角度", "右胳膊角度", "左臂高身体角度", "右臂高身体角度",
    "左肩膀胯膝盖角度", "右肩膀胯膝盖角度", "左腿角度", "右腿角度"
]

# 专业评价映射表
evaluation_map = {
    'forehand': {
        'langle': {'too_large': '左肩未打开', 'too_small': '左肩过度旋转'},
        'rangle': {'too_large': '小臂未收紧', 'too_small': '小臂外旋不够'},
        'lsangle': {'too_large': '左臂抬升过度', 'too_small': '身体前倾不足'},
        'rsangle': {'too_large': '击球点偏后', 'too_small': '击球点偏前'},
        'lhangle': {'too_large': '重心太高', 'too_small': '重心过低'},
        'rhangle': {'too_large': '右侧身体过度倾斜', 'too_small': '没有侧身'},
        'lkangle': {'too_large': '左腿过度伸展', 'too_small': '脚步未到位'},
        'rkangle': {'too_large': '右腿过度伸展', 'too_small': '脚步未到位'},
    },
    'backhand': {
        'langle': {'too_large': '小臂未收紧', 'too_small': '小臂外旋不够'},
        'rangle': {'too_large': '右肩未打开', 'too_small': '右肩过度旋转'},
        'lsangle': {'too_large': '击球点偏后', 'too_small': '击球点偏前'},
        'rsangle': {'too_large': '右臂抬升过度', 'too_small': '身体前倾不足'},
        'lhangle': {'too_large': '重心太高', 'too_small': '重心过低'},
        'rhangle': {'too_large': '右侧身体过度倾斜', 'too_small': '没有侧身'},
        'lkangle': {'too_large': '左腿过度伸展', 'too_small': '脚步未到位'},
        'rkangle': {'too_large': '右腿过度伸展', 'too_small': '脚步未到位'},
    }
}

def classify_fore_back(angle_data, svm, scaler):
    arr = np.array(angle_data)
    if arr.shape[0] < 4:
        raise ValueError("动作序列时间点不足，至少需要4个时间点")
    feat = arr[3]
    print(f"第4个时间点角度: {feat}")
    feat_scaled = scaler.transform([feat])
    return "正手" if svm.predict(feat_scaled)[0] == 0 else "反手"

def dtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2, dist=lambda x, y: norm(np.array(x) - np.array(y)))
    return distance

def score_shot(player_angles, player_accels, std_angles, std_accels):
    angle_score = 0
    for i in range(8):
        std_seq = std_angles[:, i]
        player_seq = np.array(player_angles)[:, i]
        angle_score += dtw_distance(std_seq, player_seq)

    accel_score = 0
    for i in range(3):
        std_seq = std_accels[:, i]
        player_seq = np.array(player_accels)[:, i]
        accel_score += dtw_distance(std_seq, player_seq)

    total_dtw = angle_score + accel_score
    print(f"DTW 距离: {total_dtw}")
    return round(max(0, 100 * np.exp(-total_dtw / 8000)), 2)

# ... (其他代码不变)

def predict_evaluation(svm_forehand, svm_backhand, scaler_forehand, scaler_backhand, angle_data, shot_type):
    arr = np.array(angle_data)
    if arr.shape[0] < 4:
        raise ValueError("动作序列时间点不足，至少需要4个时间点")
    feat = np.mean(arr, axis=0)
    print(f"动作角度均值: {feat}")
    svm = svm_forehand if shot_type == "正手" else svm_backhand
    scaler = scaler_forehand if shot_type == "正手" else scaler_backhand
    feat_scaled = scaler.transform([feat])
    prediction = svm.predict(feat_scaled)
    print(f"OneClassSVM 预测: {prediction}")

    labels = []
    if prediction[0] == -1:
        std_mean = std_mean_forehand if shot_type == "正手" else std_mean_backhand
        angle_threshold = 30.0
        eval_key = 'forehand' if shot_type == "正手" else 'backhand'

        player_mean = arr[3]
        print(f"第4个时间点角度（用于评价）: {player_mean}")
        for i, feature in enumerate(features):
            diff = player_mean[i] - std_mean[i]
            if abs(diff) > angle_threshold:
                direction = 'too_large' if diff > 0 else 'too_small'
                # 添加键检查，避免KeyError
                if direction in evaluation_map[eval_key][feature]:
                    labels.append(evaluation_map[eval_key][feature][direction])
                else:
                    print(f"警告: 缺失键 '{direction}' for feature '{feature}' in '{eval_key}'")
                    labels.append("未知问题")  # 默认消息，避免异常

        labels = list(dict.fromkeys(labels))

    return labels if labels else ["动作规范，技术到位"]

def process_hit(angles, accels, timestamp):
    try:
        angles = np.array(angles) % 180
        accels = np.array(accels)

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

        shot_type = classify_fore_back(angles, shot_type_clf, shot_type_scaler)
        std_angles = standard_body_angles_forehand if shot_type == "正手" else standard_body_angles_backhand
        std_accels = standard_acceleration_forehand if shot_type == "正手" else standard_acceleration_backhand

        score = score_shot(angles, accels, std_angles, std_accels)

        comments = predict_evaluation(
            oneclass_svm_forehand, oneclass_svm_backhand,
            oneclass_scaler_forehand, oneclass_scaler_backhand,
            angles, shot_type
        )

        comment_str = "，".join(comments)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{shot_id} {score} {shot_type} {comment_str} {timestamp}\n")

        print(f"击球 {shot_id}（{shot_type}）：评分 = {score}/100")
        print(f"评价：{comment_str}")
        print(f"时间戳：{timestamp}")
        print("-" * 50)

    except Exception as e:
        print(f"⚠️ 击球处理失败：{e}")
        # 修改写入格式，使其与正常一致：使用"未知"作为shot_type，错误消息作为comment
        shot_type = "未知"  # 默认shot_type
        comment_str = f"处理失败：{str(e)}"  # 错误作为comment
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{shot_id} 0.0 {shot_type} {comment_str} {timestamp}\n")

if __name__ == "__main__":
    def load_hit_moments(file_path):
        all_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = ast.literal_eval(line.strip())
                    angles = sample["pose"]
                    accels = sample["accel"]
                    timestamp = sample["timestamp"]
                    if len(np.array(angles).shape) == 1:
                        angles = [angles]
                    angles = np.array(angles) % 180
                    all_data.append((angles, accels, timestamp))
                except Exception as e:
                    print(f"❌ 解析失败：{e}")
        return all_data

    data = load_hit_moments("hit_moments.txt")
    for i, (angles, accels, timestamp) in enumerate(data, 1):
        process_hit(angles, accels, timestamp)