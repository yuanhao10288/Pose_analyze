import pandas as pd
import joblib
import numpy as np
import csv

def standard_judge(test_data):
    data_from_csv = pd.read_csv('standard_data.csv')
    features = ['langle', 'rangle', 'lsangle', 'rsangle', 
            'lhangle', 'rhangle', 'lkangle', 'rkangle']
    
    for index, row in data_from_csv.iterrows():
    # 提取当前行特征并保持二维结构
        pos_stand = row[features].values.reshape(1, -1)

    has_error = False
    analyze=""
    num_error=0
    if test_data[0][0] < pos_stand[0][0] - 15:
        analyze=analyze+"左胳膊弯曲角度不足，请增加弯曲程度。"
        # print(analyze)
        has_error = True
        num_error=num_error+1
    if test_data[0][0] > pos_stand[0][0] + 15:
        analyze=analyze+"左胳膊弯曲角度过大，请减小弯曲程度。"
        # print(analyze)
        has_error = True
        num_error=num_error+1
    if test_data[0][4] > pos_stand[0][4] + 15:
        analyze=analyze+"身体重心前倾过度，请向后退以调整重心。"
        # print(analyze)
        has_error = True
        num_error=num_error+1
    if test_data[0][4] < pos_stand[0][4] - 15:
        analyze=analyze+"身体重心过于靠后，请稍微向前倾以调整重心。"
        # print(analyze)
        has_error = True
        num_error=num_error+1
    if test_data[0][6] < pos_stand[0][6] - 15:
        analyze=analyze+"腿部弯曲程度不足，请进一步弯曲腿部，降低重心。"
        # print(analyze)
        has_error = True
        num_error=num_error+1

    # 如果没有不满足的情况，打印“完美”
    if not has_error:
        analyze="完美"
        # print("完美")

    grade=''
    if num_error==0:
        grade='A'
        # print(grade)
    elif num_error==1:
        grade='B'
        # print(grade)
    elif num_error==2:
        grade='C'
        # print(grade)
    elif num_error>=3:
        grade='D'

    return grade+" "+analyze
# 1. 加载模型和标准化器
model = joblib.load('svm_model/optimized_tennis_pose_svm.pkl')
scaler = joblib.load('svm_model/scaler.pkl')
old=""
def svm_sort(row_data):


    # print("进入函数")
    
    # 2. 读取数据
    # data_from_csv = pd.read_csv('output.csv')
    # features = ['langle', 'rangle', 'lsangle', 'rsangle', 
    #             'lhangle', 'rhangle', 'lkangle', 'rkangle']

    # 3. 逐行处理
    # amount=0
    # with conn.cursor() as cursor:
    #         # analyze 表有三个字段，分别是 id, grade, status
    #     sql = "INSERT INTO `analyze` (`id`, `grade`, `analyze`) VALUES (%s, %s, %s)"

    # skip_frames = 0
    # for index, row in data_from_csv.iterrows():
    #     if skip_frames > 0:
    #         skip_frames -= 1
    #         continue
        # 提取当前行特征并保持二维结构
    # row_data = row[features].values.reshape(1, -1)

    # 标准化
    scaled_row = scaler.transform(row_data)
    
    # 预测
    pred_class = model.predict(scaled_row)[0]
    pred_prob = model.predict_proba(scaled_row)[0]

    #输出结果
    # print(f"\nRow {index+1} 预测结果:")
    # print("预测结果：")

    if max(pred_prob) >= 0.75:
        # print(f"关键帧 - 动作类别: {pred_class}")
        # print("各类别概率:", {model.classes_[i]: f"{prob:.2%}" 
        #                     for i, prob in enumerate(pred_prob)})
        #根据预测概率评出等级
        # grade=''
        # if max(pred_prob)>=0.9:
        #     grade='A'
        #     # print(grade)
        # elif max(pred_prob)<0.9 and max(pred_prob)>=0.8:
        #     grade='B'
        #     # print(grade)
        # elif max(pred_prob)<0.8:
        #     grade='C'
        #     # print(grade)
        sortname = ''
        if pred_class == 0:
            sortname = "预备"
        elif pred_class == 1:
            sortname = "正手"
        elif pred_class == 2:
            sortname = "反手"
        else:
            return ""
        result=standard_judge(row_data)
        final=sortname+" "+result
        global old
        if old!=final:
            with open('results.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([final])
        old=final
        return final

    else:
        # print("非关键帧")
        # print("各类别概率:", {model.classes_[i]: f"{prob:.2%}" 
        #                     for i, prob in enumerate(pred_prob)})
        return ""
