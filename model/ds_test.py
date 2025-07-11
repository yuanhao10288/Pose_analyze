from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
# 配置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OPENAI_API_KEY"] = "sk-630b65b0aa5642348fe1fdb1d8ec6c96"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

VECTORDB_DIR = "deepseek_vectordb"

# 1. 强制删除旧的向量数据库（如果存在）
if os.path.exists(VECTORDB_DIR):
    # print("♻️ 正在删除旧的向量数据库...")
    shutil.rmtree(VECTORDB_DIR)

# 1. 读取本地TXT文件
txt_files = ["analyze.txt"]
docs = []
for file in txt_files:
    loader = TextLoader(file_path=file, encoding="utf-8")
    docs.extend(loader.load())

# 2. 文本分割（优化中文处理）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、"," "]
)
blocks = text_splitter.split_documents(docs)

# 3. 使用更轻量的嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 创建向量数据库
vectordb = Chroma.from_documents(
    documents=blocks,
    embedding=embedding_model,
    persist_directory="deepseek_vectordb"
)
# vectordb.persist()

# 4. 配置DeepSeek聊天模型
llm = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0.3
)

# 5. 构建问答模板
# template = """请严格根据以下上下文内容回答问题：
# 上下文：{context}
# 问题：{question}
# 回答时请使用中文，并保持客观准确："""
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


# 7. 交互式问答系统
def interactive_qa():
    print("=" * 50)
    print("知识库问答系统已启动（输入'退出'或'exit'结束问答）")
    print("=" * 50)

    while True:
        # 获取用户输入
        question = input("\n请输入问题: ").strip()

        # 退出条件
        if question.lower() in ["退出", "exit"]:
            print("\n问答结束，感谢使用！")
            break

        # 处理空输入
        if not question:
            print("问题不能为空，请重新输入")
            continue

        # 执行问答
        try:
            # response = qa_chain({"query": question})
            response = qa_chain.invoke({"query": question})
            # 输出回答
            print("\n回答：", response["result"])

            # 输出参考文本片段
            # print("\n参考的文本片段：")

            # for i, doc in enumerate(response["source_documents"], 1):
            #     print(f"[片段 {i}] 来源: {os.path.basename(doc.metadata['source'])}")
            #     print(f"内容: {doc.page_content[:150]}...\n")

        except Exception as e:
            print(f"处理问题时出错: {str(e)}")


# 启动交互式问答
if __name__ == "__main__":
    interactive_qa()