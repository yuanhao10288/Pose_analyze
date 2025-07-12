from datetime import datetime

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.json
        return jsonify({"message": "数据已接收", "received_data": data})
    else:
        return jsonify({"message": "欢迎使用 Flask API", "timestamp": str(datetime.now())})

if __name__ == '__main__':
    app.run(debug=True)