from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import requests
import tensorflow as tf

# fname = "class.txt"
# with open(fname ,"r") as f:
#     class_labels = sorted(set([word for line in f for word in line.split()]))

lungs_class_labels = ['COVID-19', 'Fibrosis', 'Tuberculosis', 'Pneumonia', 'Normal']
brain_class_labels = ['glioma', 'meningioma','notumor', 'pituitary']
eye_class_labels = ['Cataract', 'diabetic_retinopathy', 'glaucoma', 'Normal']
skin_class_labels = [
    'Melanocytic nevi (non-cancerous)',
    'Melanoma (cancerous)',
    'Benign keratosis-like lesions (non-cancerous)',
    'Basal cell carcinoma (cancerous)',
    'Actinic keratoses (non-cancerous)',
    'Vascular lesions (non-cancerous)',
    'Dermatofibroma (non-cancerous)'
]

lungs_model_path = "models/lungs_model.h5"
brain_model_path = "models/brain_tumor_model.h5"
eye_model_path = "models/eye_model.h5"
skin_model_path = "models/skin_model.h5"

def predict_class(filepath, model_path, class_labels, image_size = 299):
    np.set_printoptions(suppress=True)

    model = load_model(model_path ,compile=False)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    data = np.ndarray(shape=(1, image_size, image_size, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(filepath).convert("RGB")

    # resizing the image to be at least 299 X 299 and then cropping from the center
    size = (image_size, image_size)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_labels[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end=" \n")
    # print("Confidence Score:", confidence_score)

    result = {
        "class" : class_name,
        "score" :f'{(confidence_score*100):2.2f}%'
    }

    return result

def prepare_image(image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32)
    image /= 255.0
    image = tf.image.resize(image, [image_size, image_size])

    image = np.expand_dims(image, axis=0)

    return image

def classify_using_bytes(image_bytes, model_path, class_labels, image_size):
    model = load_model(model_path, compile=False)
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    prediction = model.predict(prepare_image(image_bytes, image_size))
    index = np.argmax(prediction, axis=1)[0]

    class_name = class_labels[index]
    confidence_score = prediction[0][index]

    return {
        'class' : class_name,
        'score' : f'{confidence_score*100:02.2f}%'
    }

def classify_using_url(url, model_path, image_size=299):
    image_source = requests.get(url).content
    
    return classify_using_bytes(image_source, model_path, image_size)

def predict(image_bytes, type):
    if type == "BRAIN":
        return classify_using_bytes(image_bytes, brain_model_path, brain_class_labels, 224)

    if type == "EYE":
        return classify_using_bytes(image_bytes, eye_model_path, eye_class_labels, 224)

    if type == "LUNGS":
        return classify_using_bytes(image_bytes, lungs_model_path, lungs_class_labels, 299)

    if type == "SKIN":
        return classify_using_bytes(image_bytes, skin_model_path, skin_class_labels, 28)

if __name__ == '__main__':
    
    print(classify_using_url(
                            url = 'https://res.cloudinary.com/ddm2qblsr/image/upload/v1690679815/COVID-8_uqezzz.png', 
                            model_path='diseases.h5'))