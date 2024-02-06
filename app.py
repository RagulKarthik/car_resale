from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
import sklearn
app = Flask(__name__,template_folder='template')

model = pickle.load(open('RandomForestRegressorModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/home')
def home():
    return render_template("index.html")
car_models = sorted(car['Name'].unique())
companies = sorted(car['Company'].unique())
locations = sorted(car['Location'].unique())
years = sorted(car['Year'].unique(), reverse=True)
seats = sorted(car['Seats'].unique())
@app.route('/info')
def info():

    return render_template("ui.html", car_models = car_models, companies = companies, locations = locations, years = years, seats = seats)

@app.route('/predict',methods=['POST','GET'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    location = request.form.get('location')
    year = int(request.form.get('year'))
    driven = int(request.form.get('driven'))
    fuel_type = request.form.get('fuel')
    transmission = request.form.get('transmission')
    owner_type = request.form.get('owner')
    mileage = float(request.form.get('mileage'))
    engine = int(request.form.get('engine'))
    power = float(request.form.get('power'))
    seat = int(request.form.get('seat'))


    prediction = model.predict(pd.DataFrame(columns=['Name', 'Company', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type',
                 'Mileage', 'Engine', 'Power', 'Seats'],
                                            data=np.array([car_model, company, location, year, driven, fuel_type, transmission, owner_type, mileage, engine, power, seat]).reshape(1, 12)))
    print(prediction)

    return render_template('ui.html', output=str(np.round(prediction[0], 2)), car_models = car_models, companies = companies, locations = locations, years = years, seats = seats)



if __name__ == '__main__':
    app.run()
