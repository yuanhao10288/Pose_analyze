import ast
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from numpy.linalg import norm
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib

# 加载正反手分类模型和标准化器
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
std_mean_forehand = np.mean(data_forehand, axis=0)  # 正手标准均值
std_mean_backhand = np.mean(data_backhand, axis=0)  # 反手标准均值
print("正手标准角度均值：", std_mean_forehand)
print("反手标准角度均值：", std_mean_backhand)

# 正手标准角度与加速度（用于DTW评分）
standard_body_angles_forehand = np.array([
    [30, 20, 110, 150, 45, 20, 10, 5],
    [35, 25, 105, 140, 50, 22, 12, 5],
    [40, 30, 100, 130, 55, 24, 15, 6],
    [60, 50, 90, 120, 60, 30, 20, 8],
    [90, 70, 80, 110, 70, 40, 25, 10],
    [70, 55, 95, 120, 65, 35, 22, 9],
    [50, 40, 100, 130, 55, 28, 18, 8],
    [40, 30, 105, 140, 50, 25, 15, 7],
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

# 反手标准角度与加速度（用于DTW评分）
standard_body_angles_backhand = np.array([
    [40, 25, 100, 140, 50, 22, 14, 6],
    [45, 30, 95, 135, 55, 24, 16, 7],
    [55, 35, 90, 125, 60, 28, 18, 8],
    [75, 55, 85, 115, 65, 32, 22, 9],
    [85, 65, 75, 105, 70, 36, 24, 10],
    [65, 50, 90, 115, 68, 33, 21, 9],
    [50, 38, 95, 125, 60, 30, 20, 8],
    [45, 35, 98, 130, 55, 28, 18, 7],
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


# 动作类型判断（使用第4个时间点的8个角度）
def classify_fore_back(angle_data, svm, scaler):
    arr = np.array(angle_data)
    if arr.shape[0] < 4:
        raise ValueError("动作序列时间点不足，至少需要4个时间点")
    feat = arr[3]  # 取第4个时间点（索引3）
    print(f"第4个时间点角度: {feat}")  # 调试：打印第4个时间点角度
    feat_scaled = scaler.transform([feat])
    return "正手" if svm.predict(feat_scaled)[0] == 0 else "反手"


# 计算 DTW 距离
def dtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2, dist=lambda x, y: norm(np.array(x) - np.array(y)))
    return distance


# 评分函数（根据正/反手标准对比）
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
    print(f"DTW 距离: {total_dtw}")  # 调试：打印 DTW 距离
    return round(max(0, 100 * np.exp(-total_dtw / 300)), 2)  # 调整衰减系数


# 使用OneClassSVM预测标签，异常评价基于训练集标准
def predict_evaluation(svm_forehand, svm_backhand, scaler_forehand, scaler_backhand, angle_data, shot_type):
    arr = np.array(angle_data)
    if arr.shape[0] < 4:
        raise ValueError("动作序列时间点不足，至少需要4个时间点")
    feat = np.mean(arr, axis=0)  # OneClassSVM仍使用整个序列均值
    print(f"动作角度均值: {feat}")  # 调试：打印角度均值
    svm = svm_forehand if shot_type == "正手" else svm_backhand
    scaler = scaler_forehand if shot_type == "正手" else scaler_backhand
    feat_scaled = scaler.transform([feat])
    prediction = svm.predict(feat_scaled)
    print(f"OneClassSVM 预测: {prediction}")  # 调试：打印预测结果

    labels = []
    if prediction[0] == -1:  # 异常动作
        std_mean = std_mean_forehand if shot_type == "正手" else std_mean_backhand
        angle_threshold = 30.0  # 阈值
        eval_key = 'forehand' if shot_type == "正手" else 'backhand'

        player_mean = arr[3]  # 使用第4个时间点的角度
        print(f"第4个时间点角度（用于评价）: {player_mean}")  # 调试
        for i, feature in enumerate(features):
            if player_mean[i] > std_mean[i] + angle_threshold:
                labels.append(evaluation_map[eval_key][feature]['too_large'])
            elif player_mean[i] < std_mean[i] - angle_threshold:
                labels.append(evaluation_map[eval_key][feature]['too_small'])

        # 去除重复评价
        labels = list(dict.fromkeys(labels))  # 保持顺序，去除重复项

    return labels if labels else ["动作规范，技术到位"]


# 处理单个击球数据
def process_hit(angles, accels):
    try:
        # 确保输入格式正确
        angles = np.array(angles) % 180  # 模180处理
        accels = np.array(accels)

        # 获取当前击球ID
        shot_id = 1
        output_file = "shot_evaluation.txt"
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip().split(" ", 1)[0]
                    shot_id = int(last_line) + 1 if last_line.isdigit() else 1

        # 分类击球类型
        shot_type = classify_fore_back(angles, shot_type_clf, shot_type_scaler)
        std_angles = standard_body_angles_forehand if shot_type == "正手" else standard_body_angles_backhand
        std_accels = standard_acceleration_forehand if shot_type == "正手" else standard_acceleration_backhand

        # 计算评分
        score = score_shot(angles, accels, std_angles, std_accels)

        # 生成评价
        comments = predict_evaluation(
            oneclass_svm_forehand, oneclass_svm_backhand,
            oneclass_scaler_forehand, oneclass_scaler_backhand,
            angles, shot_type
        )

        # 格式化评价
        comment_str = "，".join(comments)

        # 写入文件
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{shot_id} {score} {shot_type}：{comment_str}\n")

        # 控制台输出
        print(f"击球 {shot_id}（{shot_type}）：评分 = {score}/100")
        print(f"评价：{comment_str}")
        print("-" * 50)

    except Exception as e:
        print(f"⚠️ 击球处理失败：{e}")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{shot_id} 0.0 处理失败：{e}\n")


if __name__ == "__main__":
    # 批量处理（用于测试）
    def load_hit_moments(file_path):
        all_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = ast.literal_eval(line.strip())
                    angles = sample[0]
                    accels = sample[1]
                    if len(np.array(angles).shape) == 1:
                        angles = [angles]
                    angles = np.array(angles) % 180
                    all_data.append((angles, accels))
                except Exception as e:
                    print(f"❌ 解析失败：{e}")
        return all_data


    data = load_hit_moments("hit_moments.txt")
    for i, (angles, accels) in enumerate(data, 1):
        process_hit(angles, accels)