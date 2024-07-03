from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('./output/DTReg.pkl')

# Define the top features based on feature importance
top_features = ['CompetitionDistance', 'Store', 'Promo', 'CompetitionOpenSinceYear', 'DayOfWeek']

# Define all features (used during training)
all_features = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday',
                'SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Extract input values from the form
        input_features = []
        for feature in top_features:
            value = request.form[feature]
            if value == '':
                value = 0  # Default value if the input is empty
            input_features.append(float(value))
        
        # Create a DataFrame with all features, initializing with default values
        input_data = pd.DataFrame(columns=all_features)
        for feature in all_features:
            if feature in top_features:
                input_data[feature] = [float(request.form[feature]) if request.form[feature] != '' else 0]
            else:
                input_data[feature] = [0]  # or use other default values
        
        # Make prediction
        prediction = model.predict(input_data)[0]
    
    return render_template('index.html', features=top_features, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

