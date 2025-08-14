import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
import time
import logging
import csv
import math

# 获取当前文件（app.py）所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 更新 CORS 配置，添加前端运行的来源
CORS(app, resources={r"/api/*": {
    "origins": [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5000",
        "http://127.0.0.1:5000"
    ]
}})

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OPENAI_API_KEY"] = "sk-630b65b0aa5642348fe1fdb1d8ec6c96"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

VECTORDB_DIR = "deepseek_vectordb"
DATA_FILES = ["pose_data.txt", "tennis_acceleration_data.txt", "hit_moments.txt", "shot_evaluation.txt"]
ARCHIVE_DIR = "archive"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_LINES = 1000  # 最大行数

def archive_file(file_path):
    if not os.path.exists(file_path):
        logger.warning(f"文件 {file_path} 不存在，无需归档")
        return
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(ARCHIVE_DIR, f"{os.path.basename(file_path)}_{timestamp}.bak")
    shutil.move(file_path, archive_path)
    logger.info(f"文件 {file_path} 已归档至 {archive_path}")

def check_and_archive_files():
    for file_path in DATA_FILES:
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    logger.info(f"文件 {file_path} 大小 {file_size} 字节，超过限制，归档...")
                    archive_file(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = len(f.readlines())
                        if lines > MAX_LINES:
                            logger.info(f"文件 {file_path} 行数 {lines}，超过限制，归档...")
                            archive_file(file_path)
            else:
                logger.warning(f"文件 {file_path} 不存在")
        except Exception as e:
            logger.error(f"检查文件 {file_path} 时出错: {e}")

def init_qa_system():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(VECTORDB_DIR) and os.listdir(VECTORDB_DIR):
        logger.info("加载现有的向量数据库...")
        vectordb = Chroma(
            embedding_function=embedding_model,
            persist_directory=VECTORDB_DIR
        )
    else:
        logger.info("创建新的向量数据库...")
        txt_files = ["analyze.txt"]
        docs = []
        for file in txt_files:
            try:
                loader = TextLoader(file_path=file, encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                logger.error(f"加载文件 {file} 时出错: {e}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )
        blocks = text_splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            documents=blocks,
            embedding=embedding_model,
            persist_directory=VECTORDB_DIR
        )

    llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.3
    )

    template = """你的名字是由HelloWorld团队研发的网球王子，你是一个资深的网球小助手，负责对网球训练的数据分析。你的回答可以活泼一点。
    上下文：{context}
    问题：{question}
    回答时请使用中文，并保持客观准确："""
    qa_prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa_chain

qa_chain = init_qa_system()
conversation_history = []

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/ai.html')
def ai_page():
    return render_template('ai.html')

@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        logger.warning("收到空问题请求")
        return jsonify({'error': '问题不能为空'}), 400

    try:
        start_time = time.time()
        response = qa_chain.invoke({"query": question})
        answer = response["result"]
        response_time = round(time.time() - start_time, 2)

        conversation_history.append({
            'question': question,
            'answer': answer,
            'time': response_time
        })

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        logger.info(f"处理问答请求: {question}, 响应时间: {response_time}s")
        return jsonify({
            'answer': answer,
            'history': conversation_history,
            'response_time': response_time
        })
    except Exception as e:
        logger.error(f"处理问答请求失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    logger.info("历史记录已清空")
    return jsonify({'status': 'success'})

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.json
        logger.info(f"收到POST数据: {data}")
        return jsonify({"message": "数据已接收", "received_data": data})
    else:
        logger.info("处理GET请求: /api/data")
        return jsonify({"message": "欢迎使用 Flask API", "timestamp": str(datetime.now())})

@app.route('/api/evaluations', methods=['GET'])
def get_evaluations():
    try:
        check_and_archive_files()  # 检查并归档文件
        evaluations = []
        file_path = os.path.join(BASE_DIR, "shot_evaluation.txt")
        logger.debug(f"尝试读取文件: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        logger.debug("跳过空行")
                        continue
                    try:
                        parts = line.strip().split(" ", 4)  # 最多分割为5部分
                        if len(parts) != 5:
                            logger.warning(f"行格式错误: {line.strip()}")
                            continue
                        id_, score, shot_type, comments, timestamp = parts
                        evaluations.append({
                            "id": id_,
                            "score": float(score),
                            "shot_type": shot_type,
                            "comments": comments,
                            "timestamp": float(timestamp)
                        })
                    except Exception as e:
                        logger.error(f"解析评价行失败: {line.strip()}, 错误: {e}")
                        continue
            logger.info(f"读取到 {len(evaluations)} 条评价数据")
        else:
            logger.warning(f"文件 {file_path} 不存在")
        return jsonify({"evaluations": evaluations[-10:]})
    except Exception as e:
        logger.error(f"获取评价失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/training-stats', methods=['GET'])
def get_training_stats():
    try:
        file_path = os.path.join(BASE_DIR, "shot_evaluation.txt")
        total_shots = 0
        successful_shots = 0
        forehand_shots = 0
        backhand_shots = 0
        forehand_successful_shots = 0
        backhand_successful_shots = 0
        score_distribution = {
            "below_60": 0,
            "between_60_80": 0,
            "between_80_90": 0,
            "above_90": 0
        }

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        parts = line.strip().split(" ", 4)
                        if len(parts) != 5:
                            continue
                        score = float(parts[1])  # 第二个字段为评分
                        shot_type = parts[2]     # 第三个字段为击球类型
                        total_shots += 1
                        if score >= 85:
                            successful_shots += 1
                        if shot_type == "正手":
                            forehand_shots += 1
                            if score >= 85:
                                forehand_successful_shots += 1
                        elif shot_type == "反手":
                            backhand_shots += 1
                            if score >= 85:
                                backhand_successful_shots += 1
                        if score < 60:
                            score_distribution["below_60"] += 1
                        elif 60 <= score < 80:
                            score_distribution["between_60_80"] += 1
                        elif 80 <= score < 90:
                            score_distribution["between_80_90"] += 1
                        else:
                            score_distribution["above_90"] += 1
                    except Exception as e:
                        logger.error(f"解析训练统计行失败: {line.strip()}, 错误: {e}")
                        continue
            logger.info(f"训练统计：总击球数 {total_shots}, 成功击球数 {successful_shots}, "
                        f"正手 {forehand_shots}, 反手 {backhand_shots}")
        else:
            logger.warning(f"文件 {file_path} 不存在")

        success_rate = (successful_shots / total_shots * 100) if total_shots > 0 else 0
        forehand_success_rate = (forehand_successful_shots / forehand_shots * 100) if forehand_shots > 0 else 0
        backhand_success_rate = (backhand_successful_shots / backhand_shots * 100) if backhand_shots > 0 else 0
        forehand_percentage = (forehand_shots / total_shots * 100) if total_shots > 0 else 0
        backhand_percentage = (backhand_shots / total_shots * 100) if total_shots > 0 else 0

        return jsonify({
            "total_shots": total_shots,
            "success_rate": success_rate,
            "forehand_success_rate": forehand_success_rate,
            "backhand_success_rate": backhand_success_rate,
            "forehand_percentage": forehand_percentage,
            "backhand_percentage": backhand_percentage,
            "score_distribution": score_distribution
        })
    except Exception as e:
        logger.error(f"获取训练统计数据失败: {e}")
        return jsonify({
            "total_shots": 0,
            "success_rate": 0,
            "forehand_success_rate": 0,
            "backhand_success_rate": 0,
            "forehand_percentage": 0,
            "backhand_percentage": 0,
            "score_distribution": {
                "below_60": 0,
                "between_60_80": 0,
                "between_80_90": 0,
                "above_90": 0
            },
            "error": str(e)
        }), 500


@app.route('/api/download-shot-evaluation', methods=['GET'])
def download_shot_evaluation():
    try:
        file_path = os.path.join(BASE_DIR, "shot_evaluation.txt")
        logger.debug(f"文件路径: {file_path}")

        if not os.path.exists(file_path):
            logger.warning(f"文件 {file_path} 不存在")
            return jsonify({"error": "文件不存在"}), 404

        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            logger.warning(f"文件 {file_path} 为空")
            return jsonify({"error": "文件为空"}), 400

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shot_evaluation_{timestamp}.txt"

        # 直接发送文件
        return send_file(
            file_path,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    except PermissionError as e:
        logger.error(f"权限错误，无法访问文件 {file_path}: {e}")
        return jsonify({"error": "无权限访问文件，请检查文件权限"}), 403
    except UnicodeDecodeError as e:
        logger.error(f"文件编码错误 {file_path}: {e}")
        return jsonify({"error": "文件编码错误，无法读取内容"}), 500
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

@app.route('/api/track-data', methods=['GET'])
def get_track_data():
    try:
        # 假设CSV文件路径
        csv_path = os.path.join(BASE_DIR, "static", "data", "text.csv")
        times = []
        ax = []
        ay = []
        az = []

        # 读取CSV文件
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                times.append(float(row['time(ms)']))
                ax.append(float(row['ax']))
                ay.append(float(row['ay']))
                az.append(float(row['az']))

        # 计算速度（积分加速度）
        vx = [0]
        vy = [0]
        vz = [0]

        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]) / 1000.0  # 转换为秒
            vx.append(vx[i-1] + ax[i] * dt)
            vy.append(vy[i-1] + ay[i] * dt)
            vz.append(vz[i-1] + az[i] * dt)

        # 计算位置（积分速度）
        x = [0]
        y = [0]
        z = [0]

        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]) / 1000.0  # 转换为秒
            x.append(x[i-1] + vx[i] * dt)
            y.append(y[i-1] + vy[i] * dt)
            z.append(z[i-1] + vz[i] * dt)

        # 计算最大挥拍速度（单位：km/h）
        speeds = [math.sqrt(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2) for i in range(len(vx))]
        max_speed_ms = max(speeds) if speeds else 0
        max_speed_kmh = max_speed_ms * 3.6  # 转换为 km/h (1 m/s = 3.6 km/h)

        return jsonify({
            "times": times,
            "positions": {
                "x": x,
                "y": y,
                "z": z
            },
            "accelerations": {
                "ax": ax,
                "ay": ay,
                "az": az
            },
            "velocities": {
                "vx": vx,
                "vy": vy,
                "vz": vz
            },
            "max_speed": max_speed_kmh
        })
    except Exception as e:
        logger.error(f"获取轨迹数据失败: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    logger.info("启动 Flask 服务器...")
    app.run(debug=True, port=5000, host='0.0.0.0')  # 监听所有接口