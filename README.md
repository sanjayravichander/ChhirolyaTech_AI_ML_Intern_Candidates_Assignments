## Disease Symptom Prediction System

## Project Overview

The **Disease Symptom Prediction System** is a web application built using Streamlit that helps users identify potential diseases based on their symptoms. The application utilizes a machine learning model (Random Forest Classifier) trained on a dataset of symptoms and corresponding diseases. Additionally, it leverages OpenAI's language model to provide users with recommended precautions, remedies, and common over-the-counter medications for the predicted disease.

## Features

- **User-Friendly Interface**: Choose symptoms from a dropdown or describe them in your own words.
- **Disease Prediction**: Utilizes a Random Forest Classifier to predict potential diseases based on input symptoms.
- **LLM Integration**: Provides recommendations for precautions and remedies based on the predicted disease.
- **Customizable Input Method**: Users can either select symptoms from predefined lists or enter them manually.

## Installation

To run this project locally, ensure you have Python installed (preferably Python 3.7 or above). Follow the steps below to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

## Create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install the required packages:

pip install -r requirements.txt

## Set your OpenAI API Key in an environment variable:
On Windows:


## To run the Streamlit application, execute the following command in your terminal:

streamlit run app.py

## Model Training

The application is built on a dataset of diseases and symptoms. The dataset is preprocessed, and a Random Forest Classifier is trained to make predictions. The model's performance is evaluated based on its accuracy on a test set.

## Requirements
The following packages are required to run the application:

streamlit
pandas
scikit-learn
openai
sentence_transformers
You can find these in the requirements.txt file.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

### Instructions for Uploading

1. **Create a `requirements.txt` file** with the following content:
   ```plaintext
   streamlit
   pandas
   scikit-learn
   openai
   sentence_transformers

## For reference I will be attaching my streamlit application pics
![image](https://github.com/user-attachments/assets/cf68be71-733d-4400-bc96-3ac041e36d9f)
