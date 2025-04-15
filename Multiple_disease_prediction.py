# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 20:03:07 2025

@author: Aditya Raj
"""

import pickle 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved model
diabetes_bundle = pickle.load(open("E:/DATA SCIENCE/MachineLearning/diabetes_model.sav", 'rb'))
# diabetes_bundle = pickle.load(open("diabetes_model.sav", "rb")
diabetes_model = diabetes_bundle['model']
diabetes_scaler = diabetes_bundle['scaler']

heart_disease_model = pickle.load(open("E:/DATA SCIENCE/MachineLearning/heart_disease_model.sav", 'rb'))

parkinsons_model = pickle.load(open("E:/DATA SCIENCE/MachineLearning/parkinsons_model.sav", 'rb'))


st.markdown("""
    <div style='background: linear-gradient(to right, #16A085, #1ABC9C); padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;'>
        <h2 style='color: white; font-size: 1.8rem; margin: 0;'>🩺 Multiple Disease Prediction System</h2>
        <p style='color: #ecf0f1; font-size: 1rem; margin: 0.2rem;'>A Clinical Decision Support Tool for Diabetes, Heart Disease, and Parkinson’s</p>
    </div>
""", unsafe_allow_html=True)


m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1ABC9C;
        color:#ffffff;
        font-size:16px;
        padding:0.5rem 1rem;
        border-radius: 8px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #16A085;
        color:white;
    }
    </style>
""", unsafe_allow_html=True)

#sidebar for navigation

with st.sidebar:
    
    selected = option_menu(
        "Multiple Disease Prediction System",
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart-pulse', 'person'],
        default_index=0
    )

#Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    #page title 
        st.title('Diabetes Prediction Using ML')
        
        #getting the input data from the user 
        #columns for input fields 
        st.markdown("### 📋 Enter Patient Records")
        col1,col2, col3 = st.columns(3) 
        
        with col1:
            Pregnancies  = st.text_input('Number of Pregnancies')   
        with col2:
            Glucose = st.number_input('Glucose level', min_value=0, max_value=300, step=1)
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        with col2:
            Insulin = st.text_input('Insulin level')
        with col3:
            BMI = st.text_input('BMI value')  
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        with col2:
            Age = st.slider("Age", 1, 100, 25)
        
        #code for prediction 
        diab_diagnosis = ''
        st.markdown("---")
        with st.expander("🧾 Patient Record Summary"):
            st.markdown(f"""
                - **Pregnancies:** {Pregnancies}
                - **Glucose Level:** {Glucose}
                - **Blood Pressure:** {BloodPressure}
                - **Skin Thickness:** {SkinThickness}
                - **Insulin:** {Insulin}
                - **BMI:** {BMI}
                - **Diabetes Pedigree:** {DiabetesPedigreeFunction}
                - **Age:** {Age}
                """)
                
        
        
        #creating button for prediction 
        
        
        if st.button('Diabetes Test Result'):
            input_fields = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
            # Check if all inputs are filled
            if not all(input_fields):
                st.warning("⚠️ Please fill in all the fields.")
            else:
                try:
                    # ✅ Convert input values to float and reshape
                    input_data_np = np.asarray([float(x) for x in input_fields]).reshape(1, -1)
        
                    # ✅ Standardize using the same scaler used during training
                    std_input = diabetes_scaler.transform(input_data_np)
        
                    # ✅ Predict using the loaded model
                    diab_prediction = diabetes_model.predict(std_input)
        
                    # ✅ Display result
                    if diab_prediction[0] == 1:
                        st.success("🧠 The patient is likely to have Diabetes. Please refer to a specialist.")
                    else:
                        st.info("✅ The patient is unlikely to have Diabetes. No immediate concern, but consult a physician.")
        
                except ValueError:
                    st.error("❌ Invalid input. Please ensure all values are numerical.")

                # Downloadable report
                report = f"""Patient Report - Diabetes
                ---------------------------------------
                Prediction: {'Positive' if diab_prediction[0] == 1 else 'Negative'}
                Age: {Age}
                Pregnancies: {Pregnancies}
                Glucose: {Glucose}
                Blood Pressure: {BloodPressure}
                Skin Thickness: {SkinThickness}
                Insulin: {Insulin}
                BMI: {BMI}
                Diabetes Pedigree Function: {DiabetesPedigreeFunction}
                """
                st.download_button("📥 Download Diabetes Report", report, file_name="diabetes_report.txt")
                st.markdown("🔗 [CDC Diabetes Info](https://www.cdc.gov/diabetes/index.html)")
  
       
    
# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

        # page title
        st.title('Heart Disease Prediction using ML')
        st.markdown("### 📋 Enter Patient Records")
        col1, col2, col3 = st.columns(3)
    
        with col1:
            age = st.slider("Age", 1, 100, 25)
    
        with col2:
            sex = st.radio("Sex", ["Male", "Female"])
            sex = 1 if sex == "Male" else 0
    
        with col3:
            cp = st.selectbox("Chest Pain Type", ["0 - Typical", "1 - Atypical", "2 - Non-anginal", "3 - Asymptomatic"])
            cp = int(cp.split(" - ")[0])
    
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
    
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
    
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
    
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
    
        with col3:
            exang = st.text_input('Exercise Induced Angina')
    
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
    
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
    
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
    
        with col1:
            thal_option = st.selectbox(
    "Thalassemia Type",
    ["0 - Normal", "1 - Fixed Defect", "2 - Reversible Defect"]
)
            thal = int(thal_option.split(" - ")[0])

        # code for Prediction
        heart_diagnosis = ''
        
        st.markdown("---")
        with st.expander("🫀 Patient Record Summary"):
            st.markdown(f"""
            - **Age:** {age}
            - **Sex (0=Female, 1=Male):** {sex}
            - **Chest Pain Type (0-3):** {cp}
            - **Resting Blood Pressure:** {trestbps}
            - **Serum Cholestoral (mg/dl):** {chol}
            - **Fasting Blood Sugar > 120 mg/dl:** {fbs}
            - **Resting ECG Results:** {restecg}
            - **Max Heart Rate Achieved:** {thalach}
            - **Exercise Induced Angina:** {exang}
            - **ST Depression Induced by Exercise:** {oldpeak}
            - **Slope of Peak Exercise ST Segment:** {slope}
            - **Major Vessels Colored by Fluoroscopy:** {ca}
            - **Thal (0=Normal, 1=Fixed Defect, 2=Reversable Defect):** {thal}
            """)

    
        # creating a button for Prediction
    
        if st.button('Heart Disease Test Result'):
            input_fields = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
            if not all(input_fields):
                st.warning("⚠️ Please fill in all the fields.")
            else:
                user_input = [float(x) for x in input_fields]
                heart_prediction = heart_disease_model.predict([user_input])
                
                if heart_prediction[0]==1:
                    st.error("🔴 The Person is likely to have Heart Disease. Please refer specialist.")
                else:
                    st.success("🟢 The Person is not likely having Heart Disease. No immediate concern but can refer physician.")
                prediction = heart_prediction[0]

                report = f"""Patient Report - Heart Disease
                ------------------------------------------
                Prediction: {'Positive' if prediction == 1 else 'Negative'}
                Age: {age}
                Sex: {"Male" if sex == 1 else "Female"}
                Chest Pain Type: {cp}
                Resting BP: {trestbps}
                Cholesterol: {chol}
                Fasting Blood Sugar: {fbs}
                Resting ECG: {restecg}
                Max HR: {thalach}
                Exercise Induced Angina: {exang}
                Oldpeak: {oldpeak}
                Slope: {slope}
                CA (Fluoroscopy): {ca}
                Thal: {thal}
                """
                st.download_button("📥 Download Heart Report", report, file_name="heart_report.txt")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    st.title("🧠 Parkinson's Disease Prediction using Machine Learning")
    st.markdown("### 📋 Enter Patient Records")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('Fo (Hz)', 0.0, 500.0, step=0.1)
    with col2:
        fhi = st.number_input('Fhi (Hz)', 0.0, 600.0, step=0.1)
    with col3:
        flo = st.number_input('Flo (Hz)', 0.0, 400.0, step=0.1)
    with col4:
        Jitter_percent = st.number_input('Jitter (%)', 0.0, 0.2, step=0.001)
    with col5:
        Jitter_Abs = st.number_input('Jitter (Abs)', 0.0, 0.01, step=0.00001)

    with col1:
        RAP = st.number_input('RAP', 0.0, 0.02, step=0.0001)
    with col2:
        PPQ = st.number_input('PPQ', 0.0, 0.02, step=0.0001)
    with col3:
        DDP = st.number_input('DDP', 0.0, 0.04, step=0.0001)
    with col4:
        Shimmer = st.number_input('Shimmer', 0.0, 0.2, step=0.001)
    with col5:
        Shimmer_dB = st.number_input('Shimmer (dB)', 0.0, 2.0, step=0.01)

    with col1:
        APQ3 = st.number_input('APQ3', 0.0, 0.1, step=0.001)
    with col2:
        APQ5 = st.number_input('APQ5', 0.0, 0.1, step=0.001)
    with col3:
        APQ = st.number_input('APQ', 0.0, 0.2, step=0.001)
    with col4:
        DDA = st.number_input('DDA', 0.0, 0.2, step=0.001)
    with col5:
        NHR = st.number_input('NHR', 0.0, 0.2, step=0.001)

    with col1:
        HNR = st.number_input('HNR', 0.0, 40.0, step=0.1)
    with col2:
        RPDE = st.number_input('RPDE', 0.0, 1.0, step=0.01)
    with col3:
        DFA = st.number_input('DFA', 0.0, 1.0, step=0.01)
    with col4:
        spread1 = st.number_input('Spread1', -10.0, 0.0, step=0.01)
    with col5:
        spread2 = st.number_input('Spread2', 0.0, 1.0, step=0.01)

    with col1:
        D2 = st.number_input('D2', 0.0, 5.0, step=0.01)
    with col2:
        PPE = st.number_input('PPE', 0.0, 1.0, step=0.01)


    st.markdown("---")

    # Prediction button and logic
    parkinsons_diagnosis = ''
    
    
    with st.expander("🧠 Patient Record Summary"):
            st.markdown(f"""
            - **Fo (Hz):** {fo:.6f}
            - **Fhi (Hz):** {fhi:.6f}
            - **Flo (Hz):** {flo:.6f}
            - **Jitter (%):** {Jitter_percent:.6f}
            - **Jitter (Abs):** {Jitter_Abs:.6f}
            - **RAP:** {RAP:.6f}
            - **PPQ:** {PPQ:.6f}
            - **DDP:** {DDP:.6f}
            - **Shimmer:** {Shimmer:.6f}
            - **Shimmer (dB):** {Shimmer_dB:.6f}
            - **APQ3:** {APQ3:.6f}
            - **APQ5:** {APQ5:.6f}
            - **APQ:** {APQ:.6f}
            - **DDA:** {DDA:.6f}
            - **NHR:** {NHR:.6f}
            - **HNR:** {HNR:.6f}
            - **RPDE:** {RPDE:.6f}
            - **DFA:** {DFA:.6f}
            - **Spread1:** {spread1:.6f}
            - **Spread2:** {spread2:.6f}
            - **D2:** {D2:.6f}
            - **PPE:** {PPE:.6f}
            """)



    if st.button("🧪 Run Parkinson's Test"):

        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                          APQ3, APQ5, APQ, DDA, NHR, HNR,
                          RPDE, DFA, spread1, spread2, D2, PPE]

            user_input = [float(x) for x in user_input]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "🧬 **The person is likely to have Parkinson's disease.**"
            else:
                parkinsons_diagnosis = "✅ **The person is unlikely to have Parkinson's disease.**"

            st.success(parkinsons_diagnosis)
            report = f"""Patient Report - Parkinson's
            ----------------------------------------
            Prediction: {'Positive' if parkinsons_prediction[0] == 1 else 'Negative'}
            Fo: {round(fo, 6)}
            Fhi: {round(fhi, 6)}
            Flo: {round(flo, 6)}
            Jitter (%): {round(Jitter_percent, 6)}
            Jitter (Abs): {round(Jitter_Abs, 6)}
            RAP: {round(RAP, 6)}
            PPQ: {round(PPQ, 6)}
            DDP: {round(DDP, 6)}
            Shimmer: {round(Shimmer, 6)}
            Shimmer (dB): {round(Shimmer_dB, 6)}
            APQ3: {round(APQ3, 6)}
            APQ5: {round(APQ5, 6)}
            APQ: {round(APQ, 6)}
            DDA: {round(DDA, 6)}
            NHR: {round(NHR, 6)}
            HNR: {round(HNR, 6)}
            RPDE: {round(RPDE, 6)}
            DFA: {round(DFA, 6)}
            Spread1: {round(spread1, 6)}
            Spread2: {round(spread2, 6)}
            D2: {round(D2, 6)}
            PPE: {round(PPE, 6)}
            """
            st.download_button("📥 Download Parkinson's Report", report, file_name="parkinsons_report.txt")

        except ValueError:
            st.error("⚠️ Please make sure all fields are filled with valid numbers.")





st.markdown("""---""")
st.markdown(
    "Made by Aditya Raj (https://github.com/Raj-3435)"
)
