from datetime import datetime
from flask import Flask, render_template, request, jsonify
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
        file_path = "shot_evaluation.txt"
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

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    logger.info("启动 Flask 服务器...")
    app.run(debug=True, port=5000, host='0.0.0.0')  # 监听所有接口