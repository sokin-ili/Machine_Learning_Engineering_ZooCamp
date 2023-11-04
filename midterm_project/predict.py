import pickle
import numpy as np
from flask import request
from flask import Flask
from flask import jsonify

output_file = 'model_LogReg_C=0.05.bin'

with open(output_file, 'rb') as f_in:
    (sc, model) = pickle.load(f_in)

print('\n 1/3. MODEL IMPORTED! \n')

features = ['area_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'fractal_dimension_mean', 'area_se',
       'smoothness_se', 'compactness_se', 'concavity_se', 'symmetry_se',
       'fractal_dimension_se', 'texture_worst', 'area_worst',
       'smoothness_worst', 'compactness_worst', 'concavity_worst',
       'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

print('Model features:', features)

app = Flask('breast_cancer_classification')
@app.route('/predict', methods=['POST'])
def predict():
    sample = request.get_json()
    print(f'\n\n sample:{sample} \n\n')
    data = np.array([sample[key] for key in sample])
    # print(f'\n\n data: {data} \n\n')
    X = sc.transform(data.reshape(1, -1))
    # print(f'\n\n X:{X} \n\n')
    y_pred = model.predict_proba(X)[0, 1]
    cancer = y_pred >= 0.5
   

    result = {
        "cancer probability":float(np.round(y_pred,4)),
        "malignant":bool(cancer)
    }
    
    return jsonify(result)
print('\n 2/3. APP CREATED! \n')

print('\n 3/3. INITIATING THE APP! \n')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
