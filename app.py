import streamlit as st
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

st.set_page_config(
    layout="wide",
    page_title='Human Disease Detection',
)
st.title('Human Disease Detection Application')
st.write("Provide quick and accurate predictions for pneumonia, malaria, bone fractures, brain tumors, skin cancer,lung cancer ,Breast Cancer and Alzheimers detection")
st.write("*Only mammography images are allowed for breast cancer detection.")
options = ["Select One Disease", "pneumonia", "malaria", "bone fracture", "brain tumor", "skin cancer", "Lung Cancer", "Breast Cancer","Alzheimers"]
selected_option = st.selectbox("Select One Disease:", options)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

PNEUMONIA_CLASSES = ['Normal', 'Pneumonia']
MALARIA_CLASSES = ['Parasite', 'Normal']
BONE_FRACTURE_CLASSES = ['fractured', 'No fracture'] 
BRAIN_TUMOR_CLASSES = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
SKIN_CANCER_CLASSES = ["Benign", "Malignant"]
LUNG_CANCER_CLASSES = ["Benign", "Malignant", "Normal"]
BREAST_CANCER_CLASSES=["Benign with Density=1","Malignant with Density=1","Benign with Density=2","Malignant with Density=2","Benign with Density=3","Malignant with Density=3","Benign with Density=4","Malignant with Density=4"]
ALZHEIMERS_CLASSES=["Mild Demented","Moderate Demented","Non Demented","Very Mild Demented"]
# Load models once
@st.cache_resource
def load_models():
    pneumonia_model = tf.keras.models.load_model('models/pneumonia.h5')
    malaria_model = tf.keras.models.load_model('models/maleria.h5')
    bone_model = tf.keras.models.load_model('models/bonefracture.h5')
    brain_tumor_model = tf.keras.models.load_model('models/ResNet50V2_model.h5')
    skin_cancer_model = tf.keras.models.load_model('models/Skin_Cancer_Classification_with ResNet.hdf5')
    lung_cancer_model=tf.keras.models.load_model('models/lung_cancer_model.h5')
    breast_cancer_model=tf.keras.models.load_model('models/model.h5')
    alzheimers_model=tf.keras.models.load_model('models/VGG16.h5')
    return pneumonia_model, malaria_model, bone_model, brain_tumor_model, skin_cancer_model,lung_cancer_model,breast_cancer_model,alzheimers_model

PNEUMONIA_MODEL, MALARIA_MODEL, BONE_MODEL, BRAIN_TUMOR_MODEL, SKIN_CANCER_MODEL, LUNG_CANCER_MODEL,BREAST_CANCER_MODEL,ALZHEIMERS_MODEL = load_models()

def predict_image(model, image, size, grayscale=False):
    arr = img_to_array(image)
    arr = cv2.resize(arr, size[:2])  # Resize using only the first two dimensions (width, height)
    if grayscale:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        arr = np.expand_dims(arr, axis=-1)  # Add channel dimension for grayscale
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    prediction = model.predict(arr)
    return prediction

def predict_image1(model, image, size=(224,224)):
    image = image.resize(size)
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

def display_prediction(prediction, classes):
    confidence_level = round(prediction.max(), 2)
    predicted_class = classes[prediction.argmax()]
    st.write(f'Predicted Result: {predicted_class} and Confidence Level: {confidence_level}')

def bone_fracture():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        prediction = predict_image(BONE_MODEL, image, (150, 150))
        display_prediction(prediction, BONE_FRACTURE_CLASSES)

def lung_cancer():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        prediction = predict_image(LUNG_CANCER_MODEL, image, (256, 256), grayscale=True)
        display_prediction(prediction, LUNG_CANCER_CLASSES)
def breast_cancer():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        if image.mode != "RGB":
            image = image.convert("RGB")
        prediction = predict_image(BREAST_CANCER_MODEL, image, (224,224))
        display_prediction(prediction, BREAST_CANCER_CLASSES)
       
        
def pneumonia():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        prediction = predict_image(PNEUMONIA_MODEL, image, (100, 100))
        display_prediction(prediction, PNEUMONIA_CLASSES)

def malaria():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        prediction = predict_image(MALARIA_MODEL, image, (224, 224))
        display_prediction(prediction, MALARIA_CLASSES)
def alzheimers():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        if image.mode != "RGB":
            image = image.convert("RGB")
        prediction = predict_image1(ALZHEIMERS_MODEL, image, (224, 224))
        display_prediction(prediction, ALZHEIMERS_CLASSES)

def brain_tumor():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        preprocessed_image = preprocess_image(image, (224, 224))
        predictions = BRAIN_TUMOR_MODEL.predict(preprocessed_image)
        class_index = np.argmax(predictions)
        prediction_text = f"Prediction: {BRAIN_TUMOR_CLASSES[class_index]}"
        st.markdown(
            f"<div style='padding: 10px; border-radius: 5px; border: 2px solid #4CAF50; font-size: 20px; text-align: center;'>"
            f"{prediction_text}"
            f"</div>",
            unsafe_allow_html=True
        )

def skin_cancer():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, SKIN_CANCER_MODEL)
        string = f"The Skin Image that you Provided is : {SKIN_CANCER_CLASSES[np.argmax(prediction)]}"
        st.success(string)

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if __name__ == "__main__":
    if selected_option == 'pneumonia':
        pneumonia()
    elif selected_option == 'malaria':
        malaria()
    elif selected_option == 'bone fracture':
        bone_fracture()
    elif selected_option == 'brain tumor':
        brain_tumor()
    elif selected_option == 'skin cancer':
        skin_cancer()
    elif selected_option == 'Lung Cancer':
        lung_cancer()
    elif selected_option== 'Breast Cancer':
        breast_cancer()
    elif selected_option == 'Alzheimers':
        alzheimers()
    elif selected_option == 'Select One Disease':
        pass
    else:
        st.write("Something went wrong")
