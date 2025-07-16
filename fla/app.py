from datetime import datetime
from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)

# 配置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OPENAI_API_KEY"] = "sk-630b65b0aa5642348fe1fdb1d8ec6c96"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

VECTORDB_DIR = "deepseek_vectordb"


# 初始化向量数据库和问答链
def init_qa_system():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 检查向量数据库是否存在
    if os.path.exists(VECTORDB_DIR) and os.listdir(VECTORDB_DIR):
        print("加载现有的向量数据库...")
        vectordb = Chroma(
            embedding_function=embedding_model,
            persist_directory=VECTORDB_DIR
        )
    else:
        print("创建新的向量数据库...")
        # 1. 读取本地TXT文件
        txt_files = ["analyze.txt"]
        docs = []
        for file in txt_files:
            try:
                loader = TextLoader(file_path=file, encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")

        # 2. 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )
        blocks = text_splitter.split_documents(docs)

        # 创建向量数据库
        vectordb = Chroma.from_documents(
            documents=blocks,
            embedding=embedding_model,
            persist_directory=VECTORDB_DIR
        )

    # 4. 配置DeepSeek聊天模型
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.3
    )

    # 5. 构建问答模板
    template = """你的名字是由HelloWorld团队研发的网球王子，你是一个资深的网球小助手，负责对网球训练的数据分析。你的回答可以活泼一点。
    上下文：{context}
    问题：{question}
    回答时请使用中文，并保持客观准确："""
    qa_prompt = PromptTemplate.from_template(template)

    # 6. 构建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa_chain


# 初始化问答系统
qa_chain = init_qa_system()

# 存储对话历史
conversation_history = []


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/ai.html')
def ai_page():
    return render_template('ai.html')  # 确保有对应的ai.html模板文件


@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': '问题不能为空'}), 400

    try:
        # 记录开始时间
        start_time = time.time()

        # 执行问答
        response = qa_chain.invoke({"query": question})
        answer = response["result"]

        # 计算响应时间
        response_time = round(time.time() - start_time, 2)

        # 更新对话历史
        conversation_history.append({
            'question': question,
            'answer': answer,
            'time': response_time
        })

        # 限制历史记录长度
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return jsonify({
            'answer': answer,
            'history': conversation_history,
            'response_time': response_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear-history', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'success'})


@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.json
        return jsonify({"message": "数据已接收", "received_data": data})
    else:
        return jsonify({"message": "欢迎使用 Flask API", "timestamp": str(datetime.now())})


@app.route('/api/evaluations', methods=['GET'])
def get_evaluations():
    try:
        evaluations = []
        with open("shot_evaluation.txt", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(" ", 2)
                if len(parts) < 3:
                    continue
                id_, score, comments = parts
                shot_type = "正手" if "正手" in comments else "反手"
                evaluations.append({
                    "id": id_,
                    "score": score,
                    "comments": comments,
                    "shot_type": shot_type
                })
        return jsonify({"evaluations": evaluations[-10:]})  # 返回最新10条
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 创建templates文件夹（如果不存在）
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)