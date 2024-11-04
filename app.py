import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import openai
import base64
import joblib

# Load your dataset
df = pd.read_csv('C:/Users/DELL/Downloads/Final_Disease_Symptom_dataset.csv')

# Separate features and labels
symptoms = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']]
disease = df['Disease']

# Label encode each symptom column to avoid hardcoding dummy variables
label_encoders = {}
for col in symptoms.columns:
    le = LabelEncoder()
    symptoms[col] = le.fit_transform(symptoms[col])
    label_encoders[col] = le  # Store encoders for later use on new inputs

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms, disease, test_size=0.2, random_state=42)

# Train the model
model = joblib.load('C:\\Users\\DELL\\Downloads\\best_rf_model.pkl')
model.fit(X_train, y_train)

# Set up OpenAI API Key
openai.api_key = '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'

def get_llm_response(symptoms, predicted_disease):
    prompt = (
        f"The user has the following symptoms: {', '.join(symptoms)}. Based on this, the predicted disease is '{predicted_disease}'. "
        "What are the recommended precautions, remedies, and common over-the-counter medications for this condition? "
        "Please list medications that are generally considered safe for adults, unless otherwise specified, and provide any necessary precautions."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable medical assistant providing general advice."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']

# Streamlit app
st.set_page_config(page_title="Disease Symptom Prediction System", layout="wide")

# Custom CSS for background image and UI styling
with open("C:\\Users\\DELL\\Downloads\\heart-man-medicine-lungs-wallpaper-preview.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Disease Symptom Prediction System")
st.markdown("<h3 style='color: white;'>Get insights into your health!</h3>", unsafe_allow_html=True)

# Option for user to choose input method
input_method = st.radio("Choose how to enter symptoms:", ["Select from options", "Describe in your own words"], index=0)

if input_method == "Select from options":
    # User input for symptoms using select boxes
    st.subheader("Enter Symptoms")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symptom_1 = st.selectbox("Symptom 1", df['Symptom_1'].unique())
        
    with col2:
        symptom_2 = st.selectbox("Symptom 2", df['Symptom_2'].unique())
        
    with col3:
        symptom_3 = st.selectbox("Symptom 3", df['Symptom_3'].unique())
        
    with col1:
        symptom_4 = st.selectbox("Symptom 4", df['Symptom_4'].unique())
        
    with col2:
        symptom_5 = st.selectbox("Symptom 5", df['Symptom_5'].unique())
    
else:
    # User input for symptoms using text area
    st.subheader("Describe your symptoms in your own words:")
    symptoms_input = st.text_area("Symptoms Description", "")
    
# Button to predict
if st.button("Predict Disease"):
    matched_symptoms = {col: [] for col in symptoms.columns}  # Initialize matched_symptoms

    if input_method == "Select from options":
        # Encode user input symptoms using label encoders
        user_symptoms = {
            'Symptom_1': label_encoders['Symptom_1'].transform([symptom_1])[0],
            'Symptom_2': label_encoders['Symptom_2'].transform([symptom_2])[0],
            'Symptom_3': label_encoders['Symptom_3'].transform([symptom_3])[0],
            'Symptom_4': label_encoders['Symptom_4'].transform([symptom_4])[0],
            'Symptom_5': label_encoders['Symptom_5'].transform([symptom_5])[0],
        }
        
        # Convert user input to DataFrame and make prediction
        user_symptoms_df = pd.DataFrame([user_symptoms])
    
    else:
        # Process the symptoms input from text area
        input_symptoms = symptoms_input.lower().split(',')  # Split input by commas
        
        # Check for matches with existing symptoms
        for symptom in input_symptoms:
            symptom = symptom.strip()  # Remove leading/trailing spaces
            for col in symptoms.columns:
                if symptom in df[col].str.lower().values:
                    matched_symptoms[col].append(symptom)

        # Encode matched symptoms
        user_symptoms = {}
        for col in symptoms.columns:
            if matched_symptoms[col]:
                user_symptoms[col] = label_encoders[col].transform(matched_symptoms[col])[0]
            else:
                user_symptoms[col] = -1  # -1 or any other value to indicate no match

        # Convert user input to DataFrame and make prediction
        user_symptoms_df = pd.DataFrame([user_symptoms])
        st.write("Matched Symptoms:", matched_symptoms)

    # Make prediction
    predicted_disease = model.predict(user_symptoms_df)[0]
    st.success(f"Predicted Disease: {predicted_disease}")

    # Get LLM response for precautions and remedies
    all_matched_symptoms = [symptom for sublist in matched_symptoms.values() for symptom in sublist]
    precautions_and_remedies = get_llm_response(all_matched_symptoms, predicted_disease)
    st.info("Precautions and Remedies:")
    st.write(precautions_and_remedies)
