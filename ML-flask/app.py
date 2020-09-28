import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
from sklearn.preprocessing import LabelEncoder
import numpy as np
app = Flask(__name__)
model = pickle.load(open('logistic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    #[['7.0', '50.0', '1.02', '4.0', '0.0', 'normal', 'normal', 'notpresent', 'notpresent', '148.036', '18.0', '0.8', '137.528', '4.627', '11.30', '38.00', '6000.0', '4.7074', 'no', 'no', 'good', 'no', 'no']]
    
    print(x_test)
   
    if(x_test[0][5]=='normal'):
        x_test[0][5]=1.0
    else:
        x_test[0][5]=0.0
    
    if(x_test[0][6]=='normal'):
        x_test[0][6]=1.0
    else:
        x_test[0][6]=0.0
        
    if(x_test[0][7]=='present'):
        x_test[0][7]=1.0
    else:
        x_test[0][7]=0.0
        
    if(x_test[0][8]=='present'):
        x_test[0][8]=1.0
    else:
        x_test[0][8]=0.0
        
    if(x_test[0][18]=='yes'):
        x_test[0][18]=1.0
    else:
        x_test[0][18]=0.0
        
    if(x_test[0][19]=='yes'):
        x_test[0][19]=1.0
    else:
        x_test[0][19]=0.0
        
    if(x_test[0][20]=='good'):
        x_test[0][20]=0.0
    else:
        x_test[0][20]=1.0
        
    if(x_test[0][21]=='yes'):
        x_test[0][21]=1.0
    else:
        x_test[0][21]=0.0
    
    if(x_test[0][22]=='yes'):
        x_test[0][22]=1.0
    else:
        x_test[0][22]=0.0
    
    
    sc = load('rescalar.save')
        
    prediction = model.predict(sc.transform(x_test))
    print(prediction)
    output = prediction[0]
    if(output==0.):
        pred = 'has a high chance of getting Chronic Kidney Disease'
    else:
        pred = 'has a low chance of getting Chronic Kidney Disease'
    return render_template('index.html', prediction_text='The person {}'.format(pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
