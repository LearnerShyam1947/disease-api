from utils.predict import get_predicted_value, symptom_dict
from flask_restx import Api, reqparse, Resource, Namespace
from werkzeug.datastructures import FileStorage
from utils.predict_utils import predict
from flask import Flask, request
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
api = Api(
    app=app,
    title="Disease diagnosis",
    description="Get the solution for your disease",
    version="1.0",
    validate=True,
    doc="/"
)

disease_args = reqparse.RequestParser()
disease_args.add_argument(name="file", type=FileStorage, location="files", required=True)

classification_args = reqparse.RequestParser()
classification_args.add_argument(name="file", type=FileStorage, location="files", required=True)
classification_args.add_argument(
    name="type", 
    type=str, 
    location="form",
    help="select the model which you want to use for classification",
    required=True, 
    choices=[
        "EYE",
        "SKIN",
        "LUNGS", 
        "BRAIN" 
    ]
)

predict_args = reqparse.RequestParser()
predict_args.add_argument(name="symptoms", type=str, location="json", required=True)

predict_namespace = Namespace(name="predict controller", path="/predict")
disease_namespace = Namespace(name="disease controller", path="/disease")

@predict_namespace.route("/symptoms")
class Symptoms(Resource):
    def get(self):
        all_symptoms = {key: None for key in symptom_dict}
        return all_symptoms

@predict_namespace.route("/symptoms-list")
class Predict2(Resource):
    @predict_namespace.expect(predict_args)
    def post(self):
        symptoms = predict_args.parse_args()['symptoms']
        queries = [symptom.strip() for symptom in symptoms.split(',')]

        return get_predicted_value(queries)

@predict_namespace.route("/")
class Predict(Resource):
    def post(self):
        symptoms = request.json['symptoms']
        queries = list()
        for s in symptoms:
            queries.append(s['tag'])

        return get_predicted_value(queries)

@disease_namespace.route("/lungs-cancer")
class LungsDisease(Resource):
    @disease_namespace.expect(disease_args)
    def post(self):
        args = disease_args.parse_args()
        file = args['file']

        return predict(file.read(), "LUNGS")

@disease_namespace.route("/eye-diseace")
class EyeDisease(Resource):
    @disease_namespace.expect(disease_args)
    def post(self):
        args = disease_args.parse_args()
        file = args['file']

        return predict(file.read(), "EYE")

@disease_namespace.route("/skin-cancer")
class SkinDisease(Resource):
    @disease_namespace.expect(disease_args)
    def post(self):
        args = disease_args.parse_args()
        file = args['file']

        return predict(file.read(), "SKIN")

@disease_namespace.route("/brain-tumor")
class BrainTumorDisease(Resource):
    @disease_namespace.expect(disease_args)
    def post(self):
        args = disease_args.parse_args()
        file = args['file']

        return predict(file.read(), "BRAIN")

@disease_namespace.route("/general")
class AnyDisease(Resource):
    @disease_namespace.expect(classification_args)
    def post(self):
        args = classification_args.parse_args()
        file = args['file']
        type = args['type']

        return predict(file.read(), type)

api.add_namespace(predict_namespace)
api.add_namespace(disease_namespace)

if __name__ == "__main__":
    app.run(debug=True)
    

