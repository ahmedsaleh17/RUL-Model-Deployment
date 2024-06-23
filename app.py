# import our required libraries 
import pickle 
from flask import Flask, request, app, jsonify, render_template
import numpy as np 


# our API
app = Flask(__name__)

# load the model from the pickle file 
model  = pickle.load(open("pre_approach.pkl", "rb"))

# load the obd_scaler 
scaler = pickle.load(open("obd_scaler.pkl", "rb"))

# define the route for the home page 
@app.route("/")
def home_page():
    return render_template("home.html")



# define a route for the prediction 

# predict_api endpoint 
@app.route("/predict_api", methods = ["POST"])
def predict_api():

    # load the data 
    data = request.json['data']

    # get the data values 
    data_values = list(data.values())

    # convert it to numpy array 
    data_array = np.array(data_values).reshape((1,-1))

    # scale the input data
    data_scaled = scaler.transform(data_array)

    # reshape the final input data 
    input_data = data_scaled.reshape((1 ,-1)) #input (1,8)


    # prediction 
    pred = model.predict(input_data)

    prediction = round(float(pred[0]),2)

    return jsonify(prediction)




if __name__ == "__main__":
    
    # running the app
    app.run(debug= True)


    