import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import plotly.express as px
from PIL import Image
import sqlite3
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.preprocessing import image as keras_image
from keras._tf_keras.keras.models import load_model
from ultralytics import YOLO
import cv2
import pyperclip
# import pickle
# import joblib
# import textwrap
import requests

CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
REDIRECT_URI = 'http://localhost:8501/'


@st.cache_data(show_spinner=False)
def get_access_token(code):
    token_url = 'https://oauth2.googleapis.com/token'
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI
    }
    response = requests.post(token_url, data=data)
    return response.json().get('access_token')

def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS form_data
                 (id TEXT PRIMARY KEY, 
                  gender TEXT,
                  age INTEGER,
                  selected_option TEXT,
                  prediction TEXT)''')
    conn.commit()
    conn.close()

init_db()

def navigate_to(page):
    st.session_state["page"] = page
    st.rerun()

if "page" not in st.session_state:
    st.session_state["page"] = "page1"

def page1():

    st.set_page_config(
    page_title="Login",
    page_icon="welicon.png",  # Path to your favicon image
    )
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_with_google = st.button("Login with Google")

    if login_with_google:
        auth_url = f"https://accounts.google.com/o/oauth2/auth?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&response_type=code&scope=email"
        st.markdown(f"[Login with Google]({auth_url})")

    if st.button("Login"):
        if username == "Khuzaima" and password == "Deploy":
            navigate_to("page2")
        else:
            st.error("Invalid username or password")

def page2():

    st.set_page_config(
    page_title="Welcome",
    page_icon="welicon.png",  # Path to your favicon image
    )
    st.markdown("""
    <style>
    [data-testid=stAppViewContainer] {
        background-image: url('wel.jpeg');
        background-size: cover;
        text-align: center;
        border-radius: 30px;
        margin: -60px;

    }
    [data-testid=StyledLinkIconContainer]{
        margin: -40px;
        margin-left: 20px;
    }
    [data-testid=block-container]{
        padding: 44px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1>Welcome to Polyps Classifier</h1>", unsafe_allow_html=True)
    st.image('wel.png')
    if st.button("Continue"):
        navigate_to("page3")

def page3():

    st.set_page_config(
    page_title="Form",
    page_icon="welicon.png",  # Path to your favicon image
    )
    st.title("Form")
    id_input = st.text_input("Enter your ID")
    age_input = st.text_input("Enter your age")
    gender_input = st.selectbox("Select your gender", ["Male", "Female"])
    selected_option = st.selectbox("Select an option", ["Diabetic", "Non-Diabetic", "Other"])
    submit_button = st.button("Submit")

    if submit_button:
        if not (id_input.isdigit() and 13 <= len(id_input) <= 17):
            st.error("Please enter a valid ID (between 13 and 17 digits).")
        elif not (age_input.isdigit() and 0 <= int(age_input) < 130):
            st.error("Please enter a valid age.")
        else:
            conn = sqlite3.connect('data.db')
            c = conn.cursor()
            c.execute("SELECT * FROM form_data WHERE id=?", (id_input,))
            existing_data = c.fetchone()
            if existing_data:
                st.error("ID already exists in the database.")
            else:
                c.execute("INSERT INTO form_data (id, age, gender, selected_option) VALUES (?, ?, ?, ?)",
                          (id_input, age_input, gender_input, selected_option))
                conn.commit()
                conn.close()
                st.session_state.update({
                    "id_input": id_input,
                    "age_input": age_input,
                    "gender_input": gender_input,
                    "selected_option": selected_option
                })
                st.success("Form submitted successfully!")
                navigate_to("page4")

    if st.button("Back"):
        navigate_to("page2")

TARGET_WIDTH = 100
TARGET_HEIGHT = 100

model = tf.keras.models.load_model("F:/Model/8TypesResNet50.h5")

def get_class_label(predicted_class):
    if predicted_class == 0:
        return "Adenomatous"
    elif predicted_class == 1:
        return "Colon"
    elif predicted_class == 2:
        return "Esophagitis"
    elif predicted_class == 3:
        return "Hyperplastic"
    elif predicted_class == 4:
        return "Non_Polyp"
    elif predicted_class == 5:
        return "Polyp"
    elif predicted_class == 6:
        return "Serrated_Lesions"
    elif predicted_class == 7:
        return "Ulcerative_colitis"
    else:
        return "Unknown"

class_labels = {
    0: "Adenomatous",
    1: "Colon",
    2: "Esophagitis",
    3: "Hyperplastic",
    4: "Non_Polyp",
    5: "Polyp",
    6: "Serrated_Lesions",
    7: "Ulcerative_colitis"
}
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Patient Details", 0, 1, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 4, body)  # Reduced height to 8 to decrease line spacing
        self.ln()

def generate_pdf(image_path, details, filename="patient_details.pdf"):
    pdf = PDF()
    pdf.add_page()
    
    # Add image
    pdf.image(image_path, x=10, y=30, w=100)
    
    # Add details
    pdf.ln(85)
    for detail in details:
        pdf.chapter_body(detail)
    
    pdf.output(filename)



class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Patient Details", 0, 1, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

def get_class_label(predicted_class):
    return class_labels.get(predicted_class, "Unknown")

def page4():
    
    st.set_page_config(
    page_title="Classification",
    page_icon="welicon.png",  # Path to your favicon image
)

    st.title("Classify Polyps using ResNet50")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp","webp"])
        
    if uploaded_file is not None:
        # Load and preprocess the image
        img = Image.open(uploaded_file)
        img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
    
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = get_class_label(predicted_class)
    
        st.title("Details")
        # Display prediction result
        st.write("Predicted class:", predicted_label)
        st.write("ID:", st.session_state.get("id_input"))
        st.write("Gender:", st.session_state.get("gender_input"))
        st.write("Age:", st.session_state.get("age_input"))
        st.write("Selected Option:", st.session_state.get("selected_option"))

        selected_option = st.session_state["selected_option"]
        age_input = int(st.session_state["age_input"])

        if selected_option == "Non-Diabetic":
            description = "Polyps can cause specific symptoms depending on their location, such as nasal congestion, rectal bleeding, abnormal uterine bleeding, or throat discomfort. Timely medical attention is vital for accurate diagnosis and appropriate management. Treatment options are tailored to the individual's condition and the type of polyps present.\n"
        elif selected_option == "Other":
            description = "Polyps in sick patients can cause different symptoms depending on their location, such as nasal congestion, rectal bleeding, or hoarseness. Timely medical evaluation is crucial to diagnose and treat polyps effectively. Treatment options range from medications to surgical removal, tailored to the individual's condition and the type of polyps.\n"
        else:
            description = "Polyps are abnormal growths of tissue that develop inside the body. While they commonly occur in the colon, they can also appear in other areas such as the stomach, uterus, and nose.\n"

        if 10 <= age_input <= 30:
            description += " \nNot Common"
        elif 30 < age_input <= 50:
            description += " \nCommon"
        elif 50 < age_input <= 120:
            description += " \nCritical"
        else:
            description += " \nAge: **Invalid Age**"

        st.session_state["description"] = description

        st.write("Description:", description)

        # Update the database with the prediction
        try:
            conn = sqlite3.connect('data.db')
            c = conn.cursor()
            c.execute("UPDATE form_data SET prediction=? WHERE id=?",
                      (predicted_label, st.session_state.get("id_input")))
            conn.commit()
            conn.close()
            st.success("Prediction saved to database successfully!")
        except Exception as e:
            st.error(f"Error saving prediction to database: {e}")

        # Generate sample prompt after prediction
        def generate_sample_prompt(predicted_label):
            age_input = st.session_state.get("age_input", "")
            gender_input = st.session_state.get("gender_input", "")
            selected_option = st.session_state.get("selected_option", "")

            sample_prompt = f"Patient Details:\nAge: {age_input}\nGender: {gender_input}\nSelected Option: {selected_option}\nPredicted Class: {predicted_label}\n\n"
            sample_prompt += f"Please provide further details about the patient's condition or any additional information relevant to the {selected_option} patient."

            return sample_prompt

        sample_prompt = generate_sample_prompt(predicted_label)

        col1, col2 = st.columns([9, 2])
        with col1:
            st.text_area("Paste this prompt into the AI Doctor interface to obtain potential cures.", value=sample_prompt, height=200, disabled=True)
        with col2:
            if st.button("Copy"):
                pyperclip.copy(sample_prompt)
                st.success("Text Copied")

        # Generate and download PDF
        pdf_details = [
            f"ID: {st.session_state.get('id_input')}",
            f"Gender: {st.session_state.get('gender_input')}",
            f"Age: {st.session_state.get('age_input')}",
            f"Selected Option: {st.session_state.get('selected_option')}",
            f"Predicted Class: {predicted_label}",
            f"Description: {description}"
        ]
        
        if st.button("Generate PDF"):
            image_path = "uploaded_image.jpg"
            img.save(image_path)
            generate_pdf(image_path, pdf_details)
            with open("patient_details.pdf", "rb") as pdf_file:
                st.download_button(label="Download PDF", data=pdf_file, file_name="patient_details.pdf", mime="application/pdf")

    def your_main_function():
        col1, col2, col3= st.columns([3,2,7])
        with col1:
            if st.button("Chat with A.I Doctor"):
                st.markdown("[Chat with A.I Doctor](https://mediafiles.botpress.cloud/df60a97f-056e-4e5a-9b0d-469153933b98/webchat/bot.html)")
        with col2:
            if st.button("Visualize"):
                st.session_state["page"] = "page6"
                st.rerun()
        with col3:
            if st.button("Records"):
                st.session_state["page"] = "page5"
                st.rerun()
    your_main_function()

    model_cnn = tf.keras.models.load_model("F:/Model/8Types_CNN.h5")
    model_an = tf.keras.models.load_model("F:/Model/8TypesAlexNet.h5")
    model_svm = tf.keras.models.load_model("H:/Project_DL/9typesSVM.h5")
    st.caption("Apply Other Classification Algorithms")

    def detect_polyps(image, cascade_src):
        polyps_cascade = cv2.CascadeClassifier(cascade_src)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        polyps = polyps_cascade.detectMultiScale(gray, 1.5, 1)
        return polyps

    def colum():
                col1, col2, col3 = st.columns([7, 9,23])
        
                with col1:
                    if st.button("Apply CNN"):
                        # Make prediction using CNN model
                        prediction = model_cnn.predict(img_array)
                        predicted_class = np.argmax(prediction)
                        predicted_label = get_class_label(predicted_class)
                        st.write(predicted_label)
        
                with col2:
                    if st.button("Apply AlexNET"):
                        # Make prediction using AlexNET model
                        prediction = model_an.predict(img_array)
                        predicted_class = np.argmax(prediction)
                        predicted_label = get_class_label(predicted_class)
                        st.write(predicted_label)
        
                with col3:
                    if st.button("Apply SVM"):
                        # Make prediction using SVM model
                        prediction = model_svm.predict(img_array)
                        predicted_class = np.argmax(prediction)
                        predicted_label = get_class_label(predicted_class)
                        st.write(predicted_label)
        
    colum()
    st.caption("Detection Using YoloV8")
    if st.button("YoloV8"):
                st.session_state["page"] = "page7"
                st.rerun()
    if st.button("Back to Form"):
                st.session_state["page"] = "page3"
                st.rerun()

# Page to display records and delete them
def page5():

    st.set_page_config(
    page_title="Database",
    page_icon="welicon.png",  # Path to your favicon image
    )
    st.title("Database")

    # Show current records
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT * FROM form_data", conn)

    if df.empty:
        st.write("No data available.")
    else:
        st.write(df)
    
    def delete_record(record_id):
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM form_data WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()

    # Streamlit interface
    def delete_record(record_id):
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
    
        # Check if the record exists
        cursor.execute("SELECT COUNT(1) FROM form_data WHERE id = ?", (record_id,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return False
    
        # Delete the record if it exists
        cursor.execute("DELETE FROM form_data WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True
    
    # Streamlit interface
    inp = st.text_input("Enter ID to Delete Record:")
    

    col1, col2 = st.columns([2, 9])
    with col1:
        if st.button("Delete Record"):
            if inp:  # Check if the input is not empty
                try:
                    inpt = int(inp)  # Safely convert the input to an integer
                    if delete_record(inpt):
                        st.success(f"Record with ID {inpt} deleted successfully.")
                    else:
                        st.error(f"Record with ID {inpt} does not exist.")
                except ValueError:
                    st.error("Please enter a valid numeric ID with no commas(,)")
            else:
                st.error("Please enter an ID.")
        
        with col2:
         if st.button("Visualization"):
            navigate_to("page6")
            

    # Navigation buttons
    if st.button("Back"):
        navigate_to("page4")


def page6():
    
    st.set_page_config(
    page_title="Visualization",
    page_icon="welicon.png",  # Path to your favicon image
    )
    st.title("Visualization")

    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT * FROM form_data", conn)
    conn.close()

    if df.empty:
        st.write("No data available to visualize.")
    else:
        # Age Distribution Histogram
        fig_age_hist = px.histogram(df, x='age', nbins=10, title='Age Distribution')
        st.plotly_chart(fig_age_hist)

        # Age Distribution Box Plot
        fig_age_box = px.box(df, y='age', title='Age Distribution (Box Plot)')
        st.plotly_chart(fig_age_box)

        # Gender Distribution Pie Chart
        fig_gender_pie = px.pie(df, names='gender', title='Gender Distribution')
        st.plotly_chart(fig_gender_pie)

        # Polyp vs Non Polyp Distribution Bar Chart
        fig_polyp_bar = px.bar(df['selected_option'].value_counts(), x=df['selected_option'].value_counts().index, 
                               y=df['selected_option'].value_counts().values, 
                               labels={'x':'Polyp Status', 'y':'Count'}, title='Polyp vs Non Polyp Distribution')
        st.plotly_chart(fig_polyp_bar)

        # Age vs Gender Scatter Plot
        fig_age_gender = px.scatter(df, x='age', y='gender', color='selected_option', 
                                    title='Age vs Gender (Colored by Polyp Status)')
        st.plotly_chart(fig_age_gender)

        # Pairwise Scatter Plot
        fig_pairwise = px.scatter_matrix(df, dimensions=['age', 'gender'], color='prediction', 
                                         title='Pairwise Scatter Plot (Colored by Polyp Status)')
        st.plotly_chart(fig_pairwise)

    if st.button("Back"):
        navigate_to("page4")

def detect_polyps(img, cascade_src):
    # Load the Haar Cascade classifier
    polyp_cascade = cv2.CascadeClassifier(cascade_src)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect polyps in the image
    polyps = polyp_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return polyps

def page7():

    st.set_page_config(
    page_title="Detection",
    page_icon="welicon.png",  # Path to your favicon image
    )
    
    # Define the model path
    model_path = 'best.pt'
    
    # Initialize the YOLOv8 model
    model = YOLO(model_path)
    
    # Streamlit UI
    st.title('YOLOv8 Polyps Detection')
    
    st.sidebar.title('Image Config')
    # Set the default confidence to a higher value, such as 0.5
    confidence = st.sidebar.slider('Confidence', min_value=0.00, max_value=0.10, value=0.02)
    source_img = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    st.caption('Upload a photo and click the Detect Objects button to see the results.')
    
    # Display the uploaded image in the sidebar
    if source_img:
        uploaded_image = Image.open(source_img)
        st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
        # Display the uploaded image in the main page
        # st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
        if st.button('Detect Polyps'):
            # Convert the image to a format suitable for OpenCV
            image_np = np.array(uploaded_image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
            # Perform inference with YOLOv8
            results = model.predict(image_np, conf=confidence)
    
            if len(results[0].boxes) == 0:
                st.write("No objects detected.")
            else:
                annotated_image = results[0].plot()
    
                # Display the annotated image using Streamlit
                st.image(annotated_image[:, :, ::-1], caption='Detected Image', use_column_width=True)
    
                # Display detection results
                with st.expander("Detection Results"):
                    for box in results[0].boxes:
                        st.write(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xywh}")
    else:
        st.info("Please upload an image to detect objects.")



    # st.title("Polyp Detection with Haar Cascade Classifier")

    # st.write("Haar Cascade Classifier")
    # cascade_src = "H:/Machine Learning/haar - Polyps/Opencv/Khuzaima.xml"  

    # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # if uploaded_file is not None:
    #     # Convert the file to an OpenCV image.
    #     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #     img = cv2.imdecode(file_bytes, 1)

    #     # Display the uploaded image
    #     st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    #     if st.button("Detect Polyps"):
    #         polyps = detect_polyps(img, cascade_src)

    #         # Draw rectangles around the detected polyps
    #         for (x, y, w, h) in polyps:
    #             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #         # Display the result
    #         st.image(img, channels="BGR", caption="Detected Polyps", use_column_width=True)
            
    if st.button("Back"):
        navigate_to("page4")
# page7()

if __name__ == "__main__":
    if st.session_state["page"] == "page1":
        page1()
    elif st.session_state["page"] == "page2":
        page2()
    elif st.session_state["page"] == "page3":
        page3()
    elif st.session_state["page"] == "page4":
        page4()
    elif st.session_state["page"] == "page5":
        page5()
    elif st.session_state["page"] == "page6":
        page6()
    elif st.session_state["page"] == "page7":
        page7()
