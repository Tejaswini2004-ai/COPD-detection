

# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler

# # Initialize Flask app
# app = Flask(__name__)

# # Home route to serve the UI (index.html)
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route to handle prediction requests
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     # Example prediction logic (you can add your actual prediction code here)
#     features = {
#         "age": [data["age"]],
#         "smoke": [data["smoke"]],
#         "rs10007052": [data["rs10007052"]],
#         "rs8192288": [data["rs8192288"]],
#         "rs20541": [data["rs20541"]],
#         "alcoholConsumption": [data["alcoholConsumption"]],
#         "exerciseRegularly": [data["exerciseRegularly"]]
#     }
    
#     # Convert to DataFrame and make predictions
#     # Assuming the model and scaler are loaded
#     model = joblib.load('copd_model.pkl')  # Load pre-trained model
#     scaler = joblib.load('scaler.pkl')  # Load the scaler
#     X_input = pd.DataFrame(features)
#     X_input_scaled = scaler.transform(X_input)
    
#     prediction = model.predict(X_input_scaled)

#     result = {
#         "class": "COPD" if prediction[0] == 1 else "No COPD"
#     }

#     return jsonify(result)

# if __name__ == "__main__":
#     app.run(debug=True)

# # from flask import Flask, request, jsonify
# # from flask_cors import CORS  # Add this import
# # import joblib
# # import pandas as pd
# # import os

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS for all routes

# # # Load model and scaler
# # model = joblib.load(os.path.join('models', 'copd_model.pkl'))
# # scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         data = request.get_json()
# #         print("Received data:", data)  # Debugging
        
# #         features = {
# #             "age": [float(data["age"])],
# #             "smoke": [float(data["smoke"])],
# #             "rs10007052": [float(data["rs10007052"])],
# #             "rs8192288": [float(data["rs8192288"])],
# #             "rs20541": [float(data["rs20541"])],
# #             "alcoholConsumption": [float(data["alcoholConsumption"])],
# #             "exerciseRegularly": [float(data["exerciseRegularly"])]
# #         }
        
# #         X = pd.DataFrame(features)
# #         X_scaled = scaler.transform(X)
# #         prediction = model.predict(X_scaled)
        
# #         return jsonify({
# #             "prediction": int(prediction[0]),
# #             "class": "COPD" if prediction[0] == 1 else "No COPD",
# #             "status": "success"
# #         })
        
# #     except Exception as e:
# #         print("Error:", str(e))  # Debugging
# #         return jsonify({"error": str(e), "status": "failed"}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True, port=5000)





# # sanjana code
# # # app.py
# # from flask import Flask, render_template, request
# # import joblib
# # import numpy as np

# # app = Flask(__name__)
# # # model = joblib.load('copd_model.pkl')  # Load your trained ML model
# # model = joblib.load('models/copd_model.pkl')
# # scaler = joblib.load('models/scaler.pkl')

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     age = int(request.form['age'])
# #     smoking = int(request.form['smoking'])
# #     family_history = int(request.form['family_history'])
# #     breathlessness = int(request.form['breathlessness'])
# #     cough = int(request.form['cough'])

# #     input_data = np.array([[age, smoking, family_history, breathlessness, cough]])
# #     result = model.predict(input_data)

# #     if result[0] == 1:
# #         prediction = "High Risk of COPD"
# #     else:
# #         prediction = "Low Risk of COPD"

# #     return f"<h2>{prediction}</h2><br><a href='/'>Go Back</a>"

# # if __name__ == '_main_':
# #     app.run(debug=True)

# # from flask import Flask, request, jsonify, render_template
# # import joblib
# # import numpy as np

# # app = Flask(__name__)

# # model = joblib.load('models/copd_model.pkl')
# # scaler = joblib.load('models/scaler.pkl')

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()  # expect JSON
# #     print(data)  # debug

# #     # Extract fields from received JSON
# #     age = data['age']
# #     smoking = data['smoking']
# #     genetics = data['genetics']
# #     alcohol = data['alcohol']
# #     diet = data['diet']
# #     pollution = data['pollution']
    
# #     input_data = np.array([[age, smoking, genetics, alcohol, diet, pollution]])
# #     input_data_scaled = scaler.transform(input_data)
# #     prediction = model.predict(input_data_scaled)

# #     return jsonify({'prediction': int(prediction[0])})

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, request, render_template
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load model and scaler
# model = joblib.load("models/copd_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         sex = int(request.form['sex'])
#         age = int(request.form['age'])
#         bmi = float(request.form['bmi'])
#         smoke = int(request.form['smoke'])
#         rs10007052 = float(request.form['rs10007052'])
#         rs8192288 = float(request.form['rs8192288'])
#         rs20541 = float(request.form['rs20541'])
#         rs12922394 = float(request.form['rs12922394'])
#         rs2910164 = float(request.form['rs2910164'])
#         rs161976 = float(request.form['rs161976'])
#         rs473892 = float(request.form['rs473892'])
#         rs159497 = float(request.form['rs159497'])
#         rs9296092 = float(request.form['rs9296092'])

#         features = np.array([[sex, age, bmi, smoke, rs10007052, rs8192288,
#                               rs20541, rs12922394, rs2910164, rs161976,
#                               rs473892, rs159497, rs9296092]])

#         features_scaled = scaler.transform(features)
#         prediction = model.predict_proba(features_scaled)[0][1]  # Probability of class 1

#         result = f"Chances of having COPD: {prediction*100:.2f}%"
#         return render_template("index.html", prediction=result)

#     except Exception as e:
#         return f"Error: {str(e)}"

# if __name__ == '__main__':
#     app.run(debug=True)

#correct code


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np



# app = Flask(__name__)
# CORS(app)  # Allow React frontend

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     try:
#         features = [
#             int(data['sex']),
#             int(data['age']),
#             float(data['bmi']),
#             int(data['smoke']),
#             int(data['rs10007052']),
#             int(data['rs8192288']),
#             int(data['rs20541']),
#             int(data['rs12922394']),
#             int(data['rs2910164']),
#             int(data['rs161976']),
#             int(data['rs473892']),
#             int(data['rs159497']),
#             int(data['rs9296092'])
#         ]

#         prediction = 1 if sum(features) > 20 else 0
#         probability = (sum(features) / 50) * 100

#         return jsonify({
#             'prediction': prediction,
#             'probability': probability
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np

# app = Flask(__name__)
# CORS(app)  # Allow React frontend

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     try:
#         features = [
#             int(data['sex']),
#             int(data['age']),
#             float(data['bmi']),
#             int(data['smoke']),
#             int(data['rs10007052']),
#             int(data['rs8192288']),
#             int(data['rs20541']),
#             int(data['rs12922394']),
#             int(data['rs2910164']),
#             int(data['rs161976']),
#             int(data['rs473892']),
#             int(data['rs159497']),
#             int(data['rs9296092']),
#             int(data['alcohol_consumption']),
#             int(data['exercise_regularly'])
#         ]

#         # Demo prediction logic – replace with model later
#         prediction = 1 if sum(features) > 20 else 0
#         probability = (sum(features) / 75) * 100  # Adjusted for 15 features

#         return jsonify({
#             'prediction': prediction,
#             'probability': probability
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def home():
#     return "COPD Prediction API is running."

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     try:
#         features = [
#             int(data['sex']),
#             int(data['age']),
#             float(data['bmi']),
#             int(data['smoke']),
#             int(data['rs10007052']),
#             int(data['rs8192288']),
#             int(data['rs20541']),
#             int(data['rs12922394']),
#             int(data['rs2910164']),
#             int(data['rs161976']),
#             int(data['rs473892']),
#             int(data['rs159497']),
#             int(data['rs9296092']),
#             int(data['alcohol_consumption']),
#             int(data['exercise_regularly'])
#         ]

#         prediction = 'COPD' if sum(features) > 20 else 'No COPD'
#         probability = (sum(features) / 50) * 100

#         return jsonify({
#             'prediction': prediction,
#             'probability': probability
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # Load the saved model and scaler
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()  # Get the data from the request

#     # Convert the incoming JSON data to a DataFrame
#     patient_data = pd.DataFrame([data])
    
#     # Preprocess the data (apply same scaler)
#     patient_scaled = scaler.transform(patient_data)

#     # Get prediction
#     prediction = model.predict(patient_scaled)
#     prediction_prob = model.predict_proba(patient_scaled)[:, 1]  # Probability for COPD (class 1)

#     # Return the prediction and probability as JSON response
#     result = {
#         'prediction': 'COPD' if prediction[0] == 1 else 'No COPD',
#         'probability': prediction_prob[0]
#     }
    
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import joblib
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.decomposition import PCA

# app = Flask(__name__)

# # Load model and scaler
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Load transformers (used during training)
# imputer = SimpleImputer(strategy='median')
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# pca = PCA(n_components=15)

# # Dummy fit to ensure the pipeline works (will be overwritten by input)
# # Fit the transformers with sample data to make them usable
# import pandas as pd
# dummy_data = pd.read_csv('C:/Users/SRIVIDYA/Desktop/miniproject/data/finalalldata.csv')
# dummy_data = dummy_data.select_dtypes(include=['number']).drop(columns=['label'], errors='ignore')
# imputer.fit(dummy_data)
# poly.fit(imputer.transform(dummy_data))
# pca.fit(poly.transform(imputer.transform(dummy_data)))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect inputs
#         input_data = [float(request.form.get(field)) for field in request.form]

#         # Preprocess the input
#         input_array = np.array(input_data).reshape(1, -1)
#         X_imputed = imputer.transform(input_array)
#         X_poly = poly.transform(X_imputed)
#         X_pca = pca.transform(X_poly)
#         X_scaled = scaler.transform(X_pca)

#         # Make prediction
#         prediction = model.predict(X_scaled)[0]
#         probability = model.predict_proba(X_scaled)[0][1]

#         result = "YES, you have a chance of having COPD." if prediction == 1 else "NO, you don't have COPD."
#         return render_template('index.html', prediction_text=f"{result} (Confidence: {probability:.2f})")
#     except Exception as e:
#         return f"Error: {str(e)}"

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.decomposition import PCA

# app = Flask(__name__)

# # Load the trained model and scaler
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Route to serve the HTML form
# @app.route('/')
# def index():
#     return render_template('index.html')  # Make sure index.html is in the templates folder

# # Route to handle user input and make predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the user input from the form (JSON data)
#         user_input = request.get_json()

#         # Convert the input data into a DataFrame for preprocessing
#         input_data = pd.DataFrame([user_input])

#         # Preprocessing
#         imputer = SimpleImputer(strategy='median')
#         X_new = imputer.fit_transform(input_data)

#         # Feature engineering
#         poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#         X_poly = poly.fit_transform(X_new)

#         pca = PCA(n_components=15)
#         X_pca = pca.fit_transform(X_poly)

#         # Scale the data
#         X_scaled = scaler.transform(X_pca)

#         # Make predictions
#         predictions = model.predict(X_scaled)
#         probabilities = model.predict_proba(X_scaled)[:, 1]

#         # Determine the result
#         result = "YES, you have COPD" if predictions[0] == 1 else "NO, you do not have COPD"
        
#         # Send the prediction result and probability as a JSON response
#         return jsonify({
#             'result': result,
#             'probability': probabilities[0]
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.decomposition import PCA

# app = Flask(__name__)

# # Load the trained model and scaler
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Route to serve the HTML form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle user input and make predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the user input from the form (JSON data)
#         user_input = request.get_json()

#         # Convert the input data into a DataFrame for preprocessing
#         input_data = pd.DataFrame([user_input])

#         # Preprocessing
#         imputer = SimpleImputer(strategy='median')
#         X_new = imputer.fit_transform(input_data)

#         # Feature engineering
#         poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#         X_poly = poly.fit_transform(X_new)

#         pca = PCA(n_components=15)
#         X_pca = pca.fit_transform(X_poly)

#         # Scale the data
#         X_scaled = scaler.transform(X_pca)

#         # Make predictions
#         predictions = model.predict(X_scaled)
#         probabilities = model.predict_proba(X_scaled)[:, 1]

#         # Determine the result
#         result = "YES, you have COPD" if predictions[0] == 1 else "NO, you do not have COPD"
        
#         # Send the prediction result and probability as a JSON response
#         return jsonify({
#             'result': result,
#             'probability': probabilities[0]
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)




# actual one which is running



# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import joblib
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer
# import numpy as np

# app = Flask(__name__)

# # Load the trained model, scaler, and PCA
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca.pkl')

# # Features (same as the features used for training the model)
# features = ['sex', 'age', 'bmi', 'smoke', 'location', 'rs10007052', 'rs8192288', 
#             'rs20541', 'rs12922394', 'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092']

# # Preprocess input data
# def preprocess_input(input_data):
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([input_data], columns=features)

#     # Apply imputation
#     imputer = SimpleImputer(strategy='median')
#     X_new = imputer.fit_transform(input_df)

#     # Apply Polynomial Feature transformation
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X_new)

#     # Apply PCA (transform with the saved PCA)
#     X_pca = pca.transform(X_poly)

#     # Scale the input data (using the saved scaler)
#     X_scaled = scaler.transform(X_pca)
    
#     return X_scaled

# # Function to predict COPD risk
# def predict_copd(user_input):
#     # Preprocess the input data
#     X_scaled = preprocess_input(user_input)
    
#     # Predict using the trained model
#     prediction = model.predict(X_scaled)
#     probability = model.predict_proba(X_scaled)[:, 1]  # For class 1 (COPD)
    
#     # Return the result as a dictionary
#     result = {
#         "prediction": "YES, you have a risk of COPD." if prediction[0] == 1 else "NO, you don't have a risk of COPD.",
#         "probability": f"{probability[0]:.4f}"
#     }
#     return result

# # Home page route
# @app.route('/')
# def index():
#     return render_template('index.html')  # Display the form to the user

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect user input from the form
#         sex = int(request.form['sex'])
#         age = float(request.form['age'])
#         bmi = float(request.form['bmi'])
#         smoke = int(request.form['smoke'])
#         location = float(request.form['location'])
        
#         rs10007052 = float(request.form['rs10007052'])
#         rs8192288 = float(request.form['rs8192288'])
#         rs20541 = float(request.form['rs20541'])
#         rs12922394 = float(request.form['rs12922394'])
#         rs2910164 = float(request.form['rs2910164'])
#         rs161976 = float(request.form['rs161976'])
#         rs473892 = float(request.form['rs473892'])
#         rs159497 = float(request.form['rs159497'])
#         rs9296092 = float(request.form['rs9296092'])

#         # Prepare the input list
#         user_input = [sex, age, bmi, smoke, location, rs10007052, rs8192288, rs20541, 
#                       rs12922394, rs2910164, rs161976, rs473892, rs159497, rs9296092]

#         # Get prediction
#         result = predict_copd(user_input)
        
#         # Return the result as JSON
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)


# CORRECT CODE 

# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer

# app = Flask(__name__)

# # Load model, scaler, and PCA
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca.pkl')

# # Features
# features = ['sex', 'age', 'bmi', 'smoke', 'location', 'rs10007052', 'rs8192288', 
#             'rs20541', 'rs12922394', 'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092']

# # Preprocessing function
# def preprocess_input(input_data):
#     input_df = pd.DataFrame([input_data], columns=features)
#     imputer = SimpleImputer(strategy='median')
#     X_new = imputer.fit_transform(input_df)
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X_new)
#     X_pca = pca.transform(X_poly)
#     X_scaled = scaler.transform(X_pca)
#     return X_scaled

# # Predict function
# def predict_copd(user_input):
#     X_scaled = preprocess_input(user_input)
#     prediction = model.predict(X_scaled)
#     probability = model.predict_proba(X_scaled)[:, 1]
#     return {
#         "prediction": "YES, you have a risk of COPD." if prediction[0] == 1 else "NO, you don't have a risk of COPD.",
#         "probability": f"{probability[0]:.4f}"
#     }
    


# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/form')
# def form():
#     return render_template('form.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         user_input = [
#             int(request.form.get('sex')),
#             float(request.form.get('age')),
#             float(request.form.get('bmi')),
#             int(request.form.get('smoke')),
#             float(request.form.get('location')),
#             float(request.form.get('rs10007052')),
#             float(request.form.get('rs8192288')),
#             float(request.form.get('rs20541')),
#             float(request.form.get('rs12922394')),
#             float(request.form.get('rs2910164')),
#             float(request.form.get('rs161976')),
#             float(request.form.get('rs473892')),
#             float(request.form.get('rs159497')),
#             float(request.form.get('rs9296092'))
#         ]

#         print("✅ Received user input:", user_input)

#         result = predict_copd(user_input)

#         print("✅ Prediction result:", result)

#         # return render_template('result.html', prediction=result["prediction"], probability=result["probability"])
#         return render_template('result.html', result=result)
#         #return render_template('result.html', prediction=result["prediction"], probability=result["probability"])

#     except Exception as e:
#         import traceback
#         print("❌ Exception traceback:")
#         print(traceback.format_exc())
#         return f"Error: {str(e)}"


# if __name__ == '__main__':
#     app.run(debug=True)







 # ORIGINAL ONE WHICH GIVE OUTPUT 3:00


# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import joblib
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer
# import numpy as np

# app = Flask(__name__)

# # Load the trained model, scaler, and PCA
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca.pkl')

# # Features (same as the features used for training the model)
# features = ['sex', 'age', 'bmi', 'smoke', 'location', 'rs10007052', 'rs8192288', 
#             'rs20541', 'rs12922394', 'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092']

# # Preprocess input data
# def preprocess_input(input_data):
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([input_data], columns=features)

#     # Apply imputation
#     imputer = SimpleImputer(strategy='median')
#     X_new = imputer.fit_transform(input_df)

#     # Apply Polynomial Feature transformation
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X_new)

#     # Apply PCA (transform with the saved PCA)
#     X_pca = pca.transform(X_poly)

#     # Scale the input data (using the saved scaler)
#     X_scaled = scaler.transform(X_pca)
    
#     return X_scaled

# # Function to predict COPD risk
# def predict_copd(user_input):
#     # Preprocess the input data
#     X_scaled = preprocess_input(user_input)
    
#     # Predict using the trained model
#     prediction = model.predict(X_scaled)
#     probability = model.predict_proba(X_scaled)[:, 1]  # For class 1 (COPD)
    
#     # Return the result as a dictionary
#     result = {
#         "prediction": "YES, you have a risk of COPD." if prediction[0] == 1 else "NO, you don't have a risk of COPD.",
#         "probability": f"{probability[0]:.4f}"
#     }
#     return result

# # Home page route
# @app.route('/')
# def index():
#     return render_template('index.html')  # Display the form to the user
# @app.route('/form')
# def form():
#     return render_template('form.html')


# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect user input from the form
#         sex = int(request.form['sex'])
#         age = float(request.form['age'])
#         bmi = float(request.form['bmi'])
#         smoke = int(request.form['smoke'])
#         location = float(request.form['location'])
        
#         rs10007052 = float(request.form['rs10007052'])
#         rs8192288 = float(request.form['rs8192288'])
#         rs20541 = float(request.form['rs20541'])
#         rs12922394 = float(request.form['rs12922394'])
#         rs2910164 = float(request.form['rs2910164'])
#         rs161976 = float(request.form['rs161976'])
#         rs473892 = float(request.form['rs473892'])
#         rs159497 = float(request.form['rs159497'])
#         rs9296092 = float(request.form['rs9296092'])

#         # Prepare the input list
#         user_input = [sex, age, bmi, smoke, location, rs10007052, rs8192288, rs20541, 
#                       rs12922394, rs2910164, rs161976, rs473892, rs159497, rs9296092]

#         # Get prediction
#         result = predict_copd(user_input)
        
#         # Return the result as JSON
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)





# ----------------------------ACTUAL WORKING CODE -------------------------------


# import openai
# from flask import jsonify
# from flask import Flask, render_template, request, jsonify
# import openai
# import pickle
# import numpy as np


# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer
# import numpy as np

# app = Flask(__name__)

# # Load the trained model, scaler, and PCA
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca.pkl')

# # Features used during model training
# features = [
#     'sex', 'age', 'bmi', 'smoke', 'location',
#     'rs10007052', 'rs8192288', 'rs20541', 'rs12922394',
#     'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092'
# ]

# # Function to preprocess user input
# def preprocess_input(input_data):
#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_data], columns=features)

#     # Impute missing values (if any)
#     imputer = SimpleImputer(strategy='median')
#     X_new = imputer.fit_transform(input_df)

#     # Polynomial feature transformation
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X_new)

#     # PCA transformation
#     X_pca = pca.transform(X_poly)

#     # Scaling
#     X_scaled = scaler.transform(X_pca)

#     return X_scaled

# # Function to predict COPD
# def predict_copd(user_input):
#     X_scaled = preprocess_input(user_input)
#     prediction = model.predict(X_scaled)
#     probability = model.predict_proba(X_scaled)[:, 1]

#     return {
#         "prediction": "YES, you have a risk of COPD." if prediction[0] == 1 else "NO, you don't have a risk of COPD.",
#         "probability": f"{probability[0]:.4f}"
#     }

# # Route: Landing Page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route: Input Form
# @app.route('/form')
# def form():
#     return render_template('form.html')
# # Route for Support Page
# @app.route('/support')  # This is the URL for the Support page
# def support():
#     return render_template('support.html')  # Ensure the file is named 'support.html'

# @app.route('/feedback')
# def feedback():
#     return render_template('feedback.html')


# @app.route('/disease')
# def disease():
#     return render_template('disease.html')
# @app.route('/symptons')
# def symptons():
#     return render_template('symptons.html')
# @app.route('/treatment')
# def treatment():
#     return render_template('treatment.html') 
# @app.route('/prevention')
# def prevention():
#     return render_template('prevention.html')


# # Route for About/Support Page
# @app.route('/about')
# def about():
#     return render_template('about.html')  # Support page

# # Route for Contact Page
# @app.route('/contact')
# def contact():
#     return render_template('contact.html')  # Ensure this file is named 'contact.html'


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect input values
#         sex = int(request.form['sex'])
#         age = float(request.form['age'])
#         bmi = float(request.form['bmi'])
#         smoke = int(request.form['smoke'])
#         location = float(request.form['location'])

#         rs10007052 = float(request.form['rs10007052'])
#         rs8192288 = float(request.form['rs8192288'])
#         rs20541 = float(request.form['rs20541'])
#         rs12922394 = float(request.form['rs12922394'])
#         rs2910164 = float(request.form['rs2910164'])
#         rs161976 = float(request.form['rs161976'])
#         rs473892 = float(request.form['rs473892'])
#         rs159497 = float(request.form['rs159497'])
#         rs9296092 = float(request.form['rs9296092'])

#         # List of input features
#         user_input = [
#             sex, age, bmi, smoke, location,
#             rs10007052, rs8192288, rs20541, rs12922394,
#             rs2910164, rs161976, rs473892, rs159497, rs9296092
#         ]

#         # Predict
#         result = predict_copd(user_input)

#         # Render the result page
#         return render_template("result.html", result=result)

#     except Exception as e:
#         return render_template("result.html", result={
#             "prediction": "Error occurred",
#             "probability": str(e)
#         })

# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True)



# ------NICE ONE AND WORKING -------------

 
# import openai
# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer
# import os

# app = Flask(__name__)

# # Set up your OpenAI API key (preferably from environment variables)
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Or hardcode it, but avoid that in production.

# # Load the trained model, scaler, and PCA
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca.pkl')

# # Features used during model training
# features = [
#     'sex', 'age', 'bmi', 'smoke', 'location',
#     'rs10007052', 'rs8192288', 'rs20541', 'rs12922394',
#     'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092'
# ]

# # Predefined responses for the chatbot (Optional)
# predefined_qa = {
#     "What is COPD?": "Chronic Obstructive Pulmonary Disease (COPD) is a group of lung diseases that block airflow and make breathing difficult.",
#     "What are the symptoms of COPD?": "Common symptoms of COPD include shortness of breath, chronic cough, wheezing, and frequent respiratory infections.",
#     "What causes COPD?": "COPD is primarily caused by long-term exposure to harmful substances, particularly smoking. Air pollution and dust may also contribute.",
#     "How can COPD be treated?": "COPD can be managed through medications (like bronchodilators), oxygen therapy, pulmonary rehabilitation, and lifestyle changes such as quitting smoking."
# }

# # Function to preprocess user input for prediction
# def preprocess_input(input_data):
#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_data], columns=features)

#     # Impute missing values (if any)
#     imputer = SimpleImputer(strategy='median')
#     X_new = imputer.fit_transform(input_df)

#     # Polynomial feature transformation
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X_new)

#     # PCA transformation
#     X_pca = pca.transform(X_poly)

#     # Scaling
#     X_scaled = scaler.transform(X_pca)

#     return X_scaled

# # Function to predict COPD
# def predict_copd(user_input):
#     X_scaled = preprocess_input(user_input)
#     prediction = model.predict(X_scaled)
#     probability = model.predict_proba(X_scaled)[:, 1]

#     # Categorize the probability
#     risk_category = categorize_risk(probability[0])
#     precautions = get_precautions(risk_category)

#     return {
#         "prediction": "YES, you have a risk of COPD." if prediction[0] == 1 else "NO, you don't have a risk of COPD.",
#         "probability": f"{probability[0]:.4f}",
#         "risk_category": risk_category,
#         "precautions": precautions
#     }

# # Function to categorize the risk based on probability
# def categorize_risk(probability):
#     if probability < 0.17:
#         return "Very Low Risk"
#     elif 0.17 <= probability < 0.34:
#         return "Low Risk"
#     elif 0.34 <= probability < 0.51:
#         return "Moderate Risk"
#     elif 0.51 <= probability < 0.68:
#         return "High Risk"
#     elif 0.68 <= probability < 0.85:
#         return "Very High Risk"
#     else:
#         return "Extreme Risk"

# # Function to return precautions based on the risk category
# def get_precautions(risk_category):
#     precautions = {
#         "Very Low Risk": "No immediate action is needed. Maintain a healthy lifestyle with regular check-ups.",
#         "Low Risk": "Regular monitoring and lifestyle changes. Avoid smoking and limit exposure to pollutants.",
#         "Moderate Risk": "Monitor lung function and get routine check-ups. Avoid smoking, stay active, and consider using preventive medications.",
#         "High Risk": "See a doctor regularly for lung function tests. Start preventive treatments and avoid further exposure to pollutants.",
#         "Very High Risk": "Immediate medical attention is needed. Consider pulmonary rehabilitation and discuss oxygen therapy.",
#         "Extreme Risk": "Urgent medical attention and possible hospitalization. Strict adherence to medications and oxygen therapy."
#     }
#     return precautions.get(risk_category, "No precautions available.")

# # Route for the Landing Page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for the Input Form
# @app.route('/form')
# def form():
#     return render_template('form.html')

# # Route for Feedback Page
# @app.route('/feedback')
# def feedback():
#     return render_template('feedback.html')

# # Route for Disease Info Page
# @app.route('/disease')
# def disease():
#     return render_template('disease.html')

# # Route for Symptoms Info Page
# @app.route('/symptoms')
# def symptoms():
#     return render_template('symptoms.html')

# # Route for Treatment Page
# @app.route('/treatment')
# def treatment():
#     return render_template('treatment.html')

# # Route for Prevention Page
# @app.route('/prevention')
# def prevention():
#     return render_template('prevention.html')

# # Route for About Page
# @app.route('/about')
# def about():
#     return render_template('about.html')

# # Route for Contact Page
# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# # Route for the Prediction Result
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect input values from the form
#         sex = int(request.form['sex'])
#         age = float(request.form['age'])
#         bmi = float(request.form['bmi'])
#         smoke = int(request.form['smoke'])
#         location = float(request.form['location'])

#         rs10007052 = float(request.form['rs10007052'])
#         rs8192288 = float(request.form['rs8192288'])
#         rs20541 = float(request.form['rs20541'])
#         rs12922394 = float(request.form['rs12922394'])
#         rs2910164 = float(request.form['rs2910164'])
#         rs161976 = float(request.form['rs161976'])
#         rs473892 = float(request.form['rs473892'])
#         rs159497 = float(request.form['rs159497'])
#         rs9296092 = float(request.form['rs9296092'])

#         # List of input features
#         user_input = [
#             sex, age, bmi, smoke, location,
#             rs10007052, rs8192288, rs20541, rs12922394,
#             rs2910164, rs161976, rs473892, rs159497, rs9296092
#         ]

#         # Predict COPD risk
#         result = predict_copd(user_input)

#         # Render the result page with prediction, risk category, and precautions
#         return render_template("result.html", result=result)

#     except Exception as e:
#         return render_template("result.html", result={
#             "prediction": "Error occurred",
#             "probability": str(e)
#         })

# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True)



# _______NICE ONE--------------

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

# Load the trained model, scaler, and PCA
model = joblib.load('copd_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Features used during model training
features = [
    'sex', 'age', 'bmi', 'smoke', 'location',
    'rs10007052', 'rs8192288', 'rs20541', 'rs12922394',
    'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092'
]

# Predefined responses for the chatbot (Optional)
predefined_qa = {
    "What is COPD?": "Chronic Obstructive Pulmonary Disease (COPD) is a group of lung diseases that block airflow and make breathing difficult.",
    "What are the symptoms of COPD?": "Common symptoms of COPD include shortness of breath, chronic cough, wheezing, and frequent respiratory infections.",
    "What causes COPD?": "COPD is primarily caused by long-term exposure to harmful substances, particularly smoking. Air pollution and dust may also contribute.",
    "How can COPD be treated?": "COPD can be managed through medications (like bronchodilators), oxygen therapy, pulmonary rehabilitation, and lifestyle changes such as quitting smoking.",
    "How can I check my COPD risk?": "You can fill out a form with details like age, smoking status, and genetic factors to check your COPD risk."
}

# Function to preprocess user input for prediction
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    imputer = SimpleImputer(strategy='median')
    X_new = imputer.fit_transform(input_df)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_new)
    X_pca = pca.transform(X_poly)
    X_scaled = scaler.transform(X_pca)
    return X_scaled

# Function to predict COPD
def predict_copd(user_input):
    X_scaled = preprocess_input(user_input)
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1]
    risk_category = categorize_risk(probability[0])
    precautions = get_precautions(risk_category)

    return {
        "prediction": "YES, you have a risk of COPD." if prediction[0] == 1 else "NO, you don't have a risk of COPD.",
        "probability": f"{probability[0]:.4f}",
        "risk_category": risk_category,
        "precautions": precautions
    }

# Function to categorize the risk based on probability
def categorize_risk(probability):
    if probability < 0.17:
        return "Very Low Risk"
    elif 0.17 <= probability < 0.34:
        return "Low Risk"
    elif 0.34 <= probability < 0.51:
        return "Moderate Risk"
    elif 0.51 <= probability < 0.68:
        return "High Risk"
    elif 0.68 <= probability < 0.85:
        return "Very High Risk"
    else:
        return "Extreme Risk"

# Function to return precautions based on the risk category
def get_precautions(risk_category):
    precautions = {
        "Very Low Risk": "No immediate action is needed. Maintain a healthy lifestyle with regular check-ups.",
        "Low Risk": "Regular monitoring and lifestyle changes. Avoid smoking and limit exposure to pollutants.",
        "Moderate Risk": "Monitor lung function and get routine check-ups. Avoid smoking, stay active, and consider using preventive medications.",
        "High Risk": "See a doctor regularly for lung function tests. Start preventive treatments and avoid further exposure to pollutants.",
        "Very High Risk": "Immediate medical attention is needed. Consider pulmonary rehabilitation and discuss oxygen therapy.",
        "Extreme Risk": "Urgent medical attention and possible hospitalization. Strict adherence to medications and oxygen therapy."
    }
    return precautions.get(risk_category, "No precautions available.")

# Route for the Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the Input Form
@app.route('/form')
def form():
    return render_template('form.html')

# Route for the Chatbot Page
@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')



# Route for Support Page
@app.route('/support')  # This is the URL for the Support page
def support():
    return render_template('support.html')  # Ensure the file is named 'support.html'

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')


@app.route('/disease')
def disease():
    return render_template('disease.html')
@app.route('/symptons')
def symptons():
    return render_template('symptons.html')
@app.route('/treatment')
def treatment():
    return render_template('treatment.html') 
@app.route('/prevention')
def prevention():
    return render_template('prevention.html')


# Route for About/Support Page
@app.route('/about')
def about():
    return render_template('about.html')  # Support page

# Route for Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')  # Ensure this file is named 'contact.html'



#Route for the Chatbot Interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()

    # If the message matches predefined question
    response = predefined_qa.get(user_message, "Sorry, I don't have an answer to that question. Would you like to check your COPD risk instead?")

    return jsonify({'reply': response})

# Route to handle the COPD prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from the form
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        smoke = int(request.form['smoke'])
        location = float(request.form['location'])
        rs10007052 = float(request.form['rs10007052'])
        rs8192288 = float(request.form['rs8192288'])
        rs20541 = float(request.form['rs20541'])
        rs12922394 = float(request.form['rs12922394'])
        rs2910164 = float(request.form['rs2910164'])
        rs161976 = float(request.form['rs161976'])
        rs473892 = float(request.form['rs473892'])
        rs159497 = float(request.form['rs159497'])
        rs9296092 = float(request.form['rs9296092'])

        user_input = [
            sex, age, bmi, smoke, location,
            rs10007052, rs8192288, rs20541, rs12922394,
            rs2910164, rs161976, rs473892, rs159497, rs9296092
        ]

        # Predict COPD risk
        result = predict_copd(user_input)

        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html", result={
            "prediction": "Error occurred",
            "probability": str(e)
        })

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
