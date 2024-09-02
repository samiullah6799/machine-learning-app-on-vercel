import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        prediction = model.predict([[np.array(data['exp'])]])
        output = prediction[0]
        return jsonify(output)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
