import ast
import numpy as np
import pandas as pd
from .file_utils import read_file
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

diseases_list = read_file("data/diseases_list.json")
symptom_dict = read_file("data/symptom_dict.json")
# print(diseases_list)

precaution = pd.read_csv("data/precautions_df.csv")
description = pd.read_csv("data/description.csv")
medication = pd.read_csv('data/medications.csv')
workout = pd.read_csv("data/workout_df.csv")
diets = pd.read_csv('data/diets.csv')

with open("models/svm.pkl", "rb") as file:
    model = pkl.load(file)


def get_info(disease):
    descr = description[description['Disease'] == disease]['Description']
    descr = " ".join({w for w in descr})

    pre = precaution[precaution['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()[0]

    die = diets[diets['Disease'] == disease]['Diet']
    die = ast.literal_eval(die.values.tolist()[0])

    work = workout[workout['disease'] == disease]['workout']
    work = work.values.tolist()

    med = medication[medication['Disease'] == disease]['Medication']
    med = ast.literal_eval(med.values.tolist()[0])

    return {
        "description": descr,
        "precaution": pre,
        "medication": med,
        "workout": work,
        "diets": die,
    }

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptom_dict))

    for item in patient_symptoms:
        input_vector[symptom_dict[item]] = 1

    disease = diseases_list[str(model.predict([input_vector])[0])]
    print("\n", disease)

    result = get_info(disease)
    result['disease'] = disease

    return result
