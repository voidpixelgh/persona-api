from flask import Flask, request, jsonify
from flask_cors import CORS 
import time

from utils import simple_query

app = Flask(__name__)
CORS(app) 

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json 

    query = data.get('query', 'query')
    img = data.get('img', '') 

    print(img)
    print(len(img))

    if len(img) == 0:
        response = simple_query(query)

        print("returned {response}")

    return jsonify(message=f"Hello, we recieved '{query}' and img {img}!")


if __name__ == "__main__":
    app.run(debug=True)