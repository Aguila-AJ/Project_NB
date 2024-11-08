import pandas as pd
import pickle
from flask import Flask, render_template, request

# Loading the picked Standard Scaler model
with open('scaler_model.pkl', 'rb') as scaler_model: 
    loaded_scaler = pickle.load(scaler_model)
# Loading the pickled predictive model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route for prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try: 
        # Getting values of the all input variables from the form
        # Gender
        gender  = int(request.form['gender'])
        # Age 
        age = int(request.form['age'])
        # Region
        region = int(request.form['region'])
        # Previously Insured
        previously_insured = int(request.form['previously_insured'])
        # Vehicle Age
        vehicle_age = int(request.form['vehicle_age'])
        # Previous Vehicle Damage
        previous_vehicle_damage = int(request.form['previous_vehicle_damage'])
        # Annual Premium
        annual_premium = int(request.form['annual_premium'])

        # Making a DataFrame which would send to model for prediction
        # Columns name, same used in model
        columns_name = ['Gender', 'Region_Code', 'Previously_Insured', 'Vehicle_Age',
                        'Previous_Vehicle_Damage', 'Annual_Premium', 'Age_Brackets']
        # Creating a list with all values taken from user
        values_list = [gender,region, previously_insured, vehicle_age, previous_vehicle_damage, annual_premium, age]
        # Creating a Dataframe with features name and values
        test_df = pd.DataFrame([values_list], columns = columns_name)
        print("First Dataframe\n", test_df)
        # Apply same same StandardScaler Model used for transforming input data into same scale
        scaled_test_df = loaded_scaler.transform(test_df)
        print("Second Dataframe after standardization\n", scaled_test_df)
        # Result of the model
        result = model.predict(scaled_test_df)
        prediction = result[0]
        print(prediction)

        if prediction == 1:
            output = "This user will submit the Claim"
        else:
            output = "This user will not submit the Claim"

        return render_template('home.html', prediction_text = output,
                entered_values={
                'gender': 'Male' if gender == 0 else 'Female',
                'age': ["Young Adult Driver(16-24)", "Middle Aged Driver(40-64)", "Senior Driver(65+)", "Experienced Drivers(25-39)"][age],
                'region': region,
                'previously_insured': 'Yes' if previously_insured == 1 else 'No',
                'vehicle_age': ["Less than an year old", "One to two years old", "More than 2 years old"][vehicle_age],
                'vehicle_damage': 'Yes' if previous_vehicle_damage == 1 else 'No',
                'annual_premium': annual_premium
            }
            )
    
    except KeyError as e:
        return render_template('home.html', error_message=f"Missing field: {e}")

if __name__ == "__main__":
    app.run(debug=True)