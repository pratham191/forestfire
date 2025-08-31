

import pickle
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html', results=None)

@app.route('/predictdata', methods=['POST'])
def predict_datapoint(): 
    if request.method == "POST":
        # Get form values
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))   # 1 = Fire, 0 = Not Fire
        region = float(request.form.get('Region'))     # 1 = North, 0 = South

        # Scale input
        new_data_scaled = standard_scaler.transform(
            [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]
        )

        # Predict
        result = ridge_model.predict(new_data_scaled)[0]

        # Pass both result and input values back to template
        return render_template(
            'home.html',
            results=round(result, 2),
            form_data={
                'Temperature': temperature,
                'RH': rh,
                'WS': ws,
                'Rain': rain,
                'FFMC': ffmc,
                'DMC': dmc,
                'ISI': isi,
                'Classes': int(classes),
                'Region': int(region)
            }
        )

if __name__ == "__main__":
    
 app.run(host="0.0.0.0", debug=True)

