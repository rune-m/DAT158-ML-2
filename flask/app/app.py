import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import sklearn
#from featurestonumbers import FeaturesToNumbers
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load(open('models/linear_model_4.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    
    #input = pd.DataFrame()
    #input.loc[0,'budget'] = int_features[0]
    #input['popularity'] = int_features[1]
    #input['cast_size'] = int_features[2]
    #input['director'] = int_features[3]
    
    #input_prep = pipeline_func(input)
    #input_prep = input_prep.to_numpy()
    
    prediction = model.predict(final_features)
    print(prediction)

    output = round(prediction[0], 2)
    print(output)
    #output2 = round((output), 2)
    #print(output2)
    
    return render_template('index.html', prediction_text='Revenue should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    
def pipeline_func(X):
    X['budget'] = np.log1p(X['budget'])
    X['popularity'] = np.log1p(X['popularity'])
    X['cast_size'] = np.log1p(X['cast_size'])

    scaler = StandardScaler()
    scaledX = scaler.fit_transform(X)

    return X

if __name__ == "__main__":
    app.run(debug=True)