from flask import Flask, request, jsonify
import os
import pickle

app = Flask(__name__)

models = {}


# Load machine learning models
def load_models_machine_learning():
    model_files = [
        "all_model.pkl",
        "3_features/temperature_nitrate_tco2_model.pkl",
        "3_features/temperature_nitrate_phosphate_model.pkl",
        "3_features/temperature_tco2_phosphate_model.pkl",
        "3_features/nitrate_tco2_phosphate_model.pkl",
        "2_features/temperature_nitrate_model.pkl",
        "2_features/temperature_tco2_model.pkl",
        "2_features/temperature_phosphate_model.pkl",
        "2_features/nitrate_tco2_model.pkl",
        "2_features/nitrate_phosphate_model.pkl",
        "2_features/tco2_phosphate_model.pkl",
        "1_feature/temperature_model.pkl",
        "1_feature/nitrate_model.pkl",
        "1_feature/tco2_model.pkl",
        "1_feature/phosphate_model.pkl"
    ]

    for model_file in model_files:
        model_path = os.path.join("models", model_file)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_key = model_file[:-4]  # Remove the extension from the filename
        models[model_key] = model
    return models


# Load deep learning models
def load_deep_learning_models():
    model_files = [
        "all_model.pkl",
        "3_features/temperature_nitrate_tco2_model.pkl",
        "3_features/temperature_nitrate_phosphate_model.pkl",
        "3_features/temperature_tco2_phosphate_model.pkl",
        "3_features/nitrate_tco2_phosphate_model.pkl",
        "2_features/temperature_nitrate_model.pkl",
        "2_features/temperature_tco2_model.pkl",
        "2_features/temperature_phosphate_model.pkl",
        "2_features/nitrate_tco2_model.pkl",
        "2_features/nitrate_phosphate_model.pkl",
        "2_features/tco2_phosphate_model.pkl",
        "1_feature/temperature_model.pkl",
        "1_feature/nitrate_model.pkl",
        "1_feature/tco2_model.pkl",
        "1_feature/phosphate_model.pkl"
    ]

    for model_file in model_files:
        model_path = os.path.join("deepmodels", model_file)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_key = model_file[:-4]  # Remove the extension from the filename
        models[model_key] = model
    return models


# Function to predict pH value using the selected model
def predict_ph_value(models, longitude, latitude, temperature, nitrate, tco2, phosphate):
    feature_values = [temperature, nitrate, tco2, phosphate]
    num_features = sum(value is not None and value !=
                       0.0 for value in feature_values)

    if temperature != 0.0 and nitrate != 0.0 and tco2 != 0.0 and phosphate != 0.0:
        model_key = "all_model"
    elif nitrate != 0.0 and tco2 != 0.0 and phosphate != 0.0:
        model_key = "3_features/nitrate_tco2_phosphate_model"
    elif temperature != 0.0 and nitrate != 0.0 and phosphate != 0.0:
        model_key = "3_features/temperature_nitrate_phosphate_model"
    elif temperature != 0.0 and tco2 != 0.0 and phosphate != 0.0:
        model_key = "3_features/temperature_tco2_phosphate_model"
    elif temperature != 0.0 and nitrate != 0.0 and tco2 != 0.0:
        model_key = "3_features/temperature_nitrate_tco2_model"
    elif nitrate != 0.0 and phosphate != 0.0:
        model_key = "2_features/nitrate_phosphate_model"
    elif nitrate != 0.0 and tco2 != 0.0:
        model_key = "2_features/nitrate_tco2_model"
    elif tco2 != 0.0 and phosphate != 0.0:
        model_key = "2_features/tco2_phosphate_model"
    elif temperature != 0.0 and nitrate != 0.0:
        model_key = "2_features/temperature_nitrate_model"
    elif temperature != 0.0 and tco2 != 0.0:
        model_key = "2_features/temperature_tco2_model"
    elif temperature != 0.0 and phosphate != 0.0:
        model_key = "2_features/temperature_phosphate_model"
    elif temperature != 0.0:
        model_key = "1_feature/temperature_model"
    elif nitrate != 0.0:
        model_key = "1_feature/nitrate_model"
    elif tco2 != 0.0:
        model_key = "1_feature/tco2_model"
    elif phosphate != 0.0:
        model_key = "1_feature/phosphate_model"
    else:
        return jsonify({"error": "Please enter at least one feature value."})

    if model_key not in models:
        return jsonify({"error": f"No model found for key: {model_key}"})

    model = models[model_key]
    prediction = model.predict([feature_values[:num_features]])
    return jsonify({"prediction": str(prediction)})


@app.route("/")
def demo():
    return "Hello World"
# API endpoint for pH prediction


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    longitude = data.get('longitude')
    latitude = data.get('latitude')
    temperature = data.get('temperature')
    nitrate = data.get('nitrate')
    tco2 = data.get('tco2')
    phosphate = data.get('phosphate')
    model_type = data.get('model_type')

    if None in [longitude, latitude, temperature, nitrate, tco2, phosphate, model_type]:
        return jsonify({"error": "Missing required parameters."})

    if model_type == "machine_learning":
        models = load_models_machine_learning()

    elif model_type == "deep_learning":
        models = load_deep_learning_models()
    else:
        return jsonify({"error": "Invalid model type."})

    return predict_ph_value(models, longitude, latitude, temperature, nitrate, tco2, phosphate)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
