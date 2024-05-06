import streamlit as st
import base64

# Title
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")
st.markdown('<style>div.block-container{padding-top:3rem; color: white;font-weight:bold;}</style>', unsafe_allow_html=True)


# # Page Layout
# c1, c2 = st.columns(2)
# with c1:
#     st.image('../img/cittabase_logo_bg.jpg', width=200)

# Heading
rw1, rw2, rw3 = st.columns([30,55, 15])
with rw2:
  # hd = st.title(":fuelpump: Drilling Analysis Dashboard")
  st.markdown("<h1 style='color: white;font-weight:bold;text-align:center;'>Heart Disease Analysis Dashboard</h1>", unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    
    [data-testid="stWidgetLabel"] {{
 		color: white;
    }}
    
    </style>
    """,
    unsafe_allow_html=True
    )
bg_img = add_bg_from_local('/mount/src/heart-disease-detection/heart.jpeg')



# Introduction
st.markdown("<h3 style='color: white;font-weight:bold;'>Introduction</h3>", unsafe_allow_html=True)

# st.write(':oil_drum: The data combines real-time drilling data with **Computed Petrophysical Output** (CPO) log data from Oil Field in the ***North Sea***. It encompasses various parameters crucial for understanding drilling operations and petrophysical characteristics:')

st.markdown(f"""
                <p style="color: black; font-size: 18px;font-weight:bold;">
                  <i>Heart disease</i> is one of the leading causes of death worldwide, affecting millions of people each year. Early detection and timely intervention are crucial in managing and preventing heart-related issues. With advancements in technology and data analytics, it is now possible to leverage data-driven approaches for heart disease detection and prediction.
                </p>
        """, unsafe_allow_html=True)

st.markdown("<h3 style='color: white;font-weight:bold;'>Use Case</h3>", unsafe_allow_html=True)

st.markdown(f"""
<ul style="color: black;font-weight:bold;">
  <li style="color: black;font-weight:bold;font-size: 18px;">
    <p style="color: white;font-weight:bold;font-size: 24px;">Early Detection:</p> The dashboard can help healthcare professionals and individuals assess the risk of heart disease based on various factors such as age, gender, blood pressure, cholesterol levels, and lifestyle habits. By identifying high-risk individuals early, preventive measures can be implemented to reduce the likelihood of heart-related complications.
  </li>
  <li style="color: black;font-weight:bold;font-size: 18px;">
    <p style="color: white;font-weight:bold;font-size: 24px;">Risk Assessment:</p> Through data analytics and predictive modeling, the dashboard can provide personalized risk assessments for individuals. By analyzing historical health data and lifestyle factors, the dashboard can predict the likelihood of developing heart disease in the future. This information can empower individuals to make informed decisions about their health and lifestyle choices.
  </li>
  <li style="color: black;font-weight:bold;font-size: 18px;">
    <p style="color: white;font-weight:bold;font-size: 24px;">Treatment Planning:</p> For patients already diagnosed with heart disease, the dashboard can assist healthcare providers in developing personalized treatment plans. By analyzing patient data and medical history, the dashboard can recommend appropriate interventions, medications, and lifestyle modifications to manage the condition effectively.
  </li>
  <li style="color: black;font-weight:bold;font-size: 18px;">
    <p style="color: white;font-weight:bold;font-size: 24px;">Research and Insights:</p> The dashboard can also serve as a valuable tool for researchers and healthcare organizations to gain insights into heart disease trends, risk factors, and treatment outcomes. By analyzing aggregated data from diverse sources, researchers can identify patterns, correlations, and emerging trends in heart disease prevalence and management.
  </li>
</ul>
""", unsafe_allow_html=True)

st.markdown("<h3 style='color: white;font-weight:bold;'>Conclusion</h3>", unsafe_allow_html=True)
st.markdown(f"""
                <p style="color: black; font-size: 18px;font-weight:bold;">
                    Overall, the Heart Disease Detection Analytic Dashboard and Prediction aims to harness the power of data analytics to improve early detection, risk assessment, treatment planning, and research in the field of cardiovascular health. By providing actionable insights and personalized recommendations, the dashboard has the potential to make a significant impact on reducing the burden of heart disease and improving overall patient outcomes.
                </p>
        """, unsafe_allow_html=True)
