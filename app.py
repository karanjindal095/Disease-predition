import pickle
import numpy as np
from flask import Flask, render_template, request

# Global variables
app = Flask(__name__)
loadedmodel = pickle.load(open('project.pkl', 'rb'))

disease_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches', 'continuous_sneezing', 'shivering',
    'chills', 'watering_from_eyes', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough',
    'chest_pain', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes',
    'burning_micturition', 'spotting_urination', 'passage_of_gases', 'internal_itching', 'indigestion',
    'muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts', 'fatigue', 'weight_loss',
    'restlessness', 'lethargy', 'irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity',
    'excessive_hunger', 'increased_appetite', 'polyuria', 'sunken_eyes', 'dehydration', 'diarrhoea',
    'breathlessness', 'family_history', 'mucoid_sputum', 'headache', 'dizziness', 'loss_of_balance',
    'lack_of_concentration', 'stiff_neck', 'depression', 'irritability', 'visual_disturbances', 'back_pain',
    'weakness_in_limbs', 'neck_pain', 'weakness_of_one_body_side', 'altered_sensorium', 'dark_urine',
    'sweating', 'muscle_pain', 'mild_fever', 'swelled_lymph_nodes', 'malaise', 'red_spots_over_body',
    'joint_pain', 'pain_behind_the_eyes', 'constipation', 'toxic_look_typhos', 'belly_pain', 'yellow_urine',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'acute_liver_failure', 'swelling_of_stomach', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'fluid_overload', 'phlegm', 'blood_in_sputum', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'loss_of_smell', 'fast_heart_rate', 'rusty_sputum', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'cramps', 'bruising', 'swollen_legs',
    'swollen_blood_vessels', 'prominent_veins_on_calf', 'weight_gain', 'cold_hands_and_feets', 'mood_swings',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremities', 'abnormal_menstruation',
    'muscle_weakness', 'anxiety', 'slurred_speech', 'palpitations', 'drying_and_tingling_lips', 'knee_pain',
    'hip_joint_pain', 'swelling_joints', 'painful_walking', 'movement_stiffness', 'spinning_movements',
    'unsteadiness', 'pus_filled_pimples', 'blackheads', 'scurrying', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'Disease'
]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    user_symptoms = [request.form[symptom] for symptom in disease_symptoms[:-1]]
    
    # Convert symptom data to numeric format using one-hot encoding
    encoded_symptoms = np.zeros(len(disease_symptoms) - 1)
    for i, symptom in enumerate(user_symptoms):
        if symptom == '1':
            encoded_symptoms[i] = 1
    
    # Predict disease using the loaded model
    Disease = loadedmodel.predict([encoded_symptoms])[0]

    return render_template('result.html', output=Disease)

# Main function
if __name__ == '__main__':
    app.run(debug=True)
