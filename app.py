from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.carPricePrediction.pipeline.prediction import PredictionPipeline
from src.carPricePrediction.components.data_transformation import TargetEncodingTransformer
import joblib



app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:



            insurance_validity= str(request.form.get('insurance_validity')),
            fuel_type= str(request.form.get('fuelType')),
            ownership= str(request.form.get('ownership')),
            transmission= str(request.form.get('transmission')),
            short_carname= str(request.form.get('short_carname')),
            manufacturing_year= int(request.form.get('manufacturing_year')),
            seats= int(request.form.get('seats')),
            kms_driven= int(request.form.get('kms_driven')),
            mileage_kmpl= int(request.form.get('mileage_kmpl')),
            engine_cc= int(request.form.get('engine_cc')),
            torque_nm= int(request.form.get('torque_nm'))

            # Include rowNumber and customerId for data schema sake
            car_name,registration_year = 0, 0


            
            field_names = [
                'insurance_validity','fuelType','ownership','transmission','short_carname','manufacturing_year','seats','kms_driven','mileage_kmpl','engine_cc','torque_nm','car_name','registration_year',
            ]

            # Creating a data array
            data = [
                insurance_validity,fuel_type,ownership,transmission,short_carname,manufacturing_year,seats,kms_driven,mileage_kmpl,engine_cc,torque_nm,car_name,registration_year,
            ]

            data = [item[0] if isinstance(item, tuple) else item for item in data]


            # print("printing data",data)
            matrix = np.array(data).reshape(1,-1)
            

            data = pd.DataFrame(matrix, columns=field_names)

            # data=target_encode.transform(data)

            print(data)

            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(data)

            print(prediction)

            return render_template('result.html', prediction=prediction)
        except Exception as e:
            print("errorrr: ",e)
            return str(e)

    print("retunrfuibdiuwedbvwediouvnwedo")
    return render_template('index.html')



if __name__ == "__main__":
	app.run(debug=True)