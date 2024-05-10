import streamlit as st
import pandas as pd 
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import plotly.express as px


# Page Title
st.set_page_config(layout="wide",page_title="Heart Disease Prediction")


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
    
    [data-testid="stFileUploader"] {{
        width: 400px;
    }}
    
    </style>
    """,
    unsafe_allow_html=True
    )
bg_img = add_bg_from_local('/mount/src/heart-disease-detection/heart.jpeg')


head_1, head_2, head_3 = st.columns([20,60, 20])
        
with head_2:
    st.markdown("<h1 style='color:white;text-align:center;'>Cardio Prediction for Heart Disease Detection</h1>", unsafe_allow_html=True)
    
# For Blank Space
blank1, blank2 = st.columns(2)
blank3, blank4 = st.columns(2)
blank4, blank6 = st.columns(2)


# Training Data
trainData = pd.read_csv('/mount/src/heart-disease-detection/heart_dis.csv')
trainDf = pd.DataFrame(trainData)

# st.write(trainDf)


# Splitting features and targets
X_features = trainDf.drop(columns=['Heart Disease'])
Y_target = trainDf['Heart Disease']
# st.write(Y_train)


# Splitting Trainging data
x_train, x_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=0)

# Testing Data
# testData = pd.read_csv('Heart_dis_detect\dataset\dataTest.csv')
# testDf = pd.DataFrame(testData)

# Splitting features
# X_test = testDf


## Model Selection
# Heading
st.markdown("<h2 style='color: white;'><b>Data Accuracy</b></h2>", unsafe_allow_html=True)

filter_option = ('Decision Trees Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier')
        
col1, col2 = st.columns([0.2, 0.8])
with col1:
    mod_sel = st.selectbox("***Model Selection***", filter_option)

if mod_sel == 'Decision Trees Classifier':
    # Standardization for feature scaling
    pipe = Pipeline([("std_scalar", StandardScaler()), ("Decision_tree", DecisionTreeClassifier())]) 
elif mod_sel == 'Random Forest Classifier':
    # Standardization for feature scaling
    pipe = Pipeline([("std_scalar", StandardScaler()), ("randomForest_tree", RandomForestClassifier())]) 
elif mod_sel == 'Gradient Boosting Classifier':
    # Standardization for feature scaling
    pipe = Pipeline([("std_scalar", StandardScaler()), ("gradient_boosting", GradientBoostingClassifier())]) 


# fitting model
pipe.fit(x_train, y_train)


# model prediction
pred = pipe.predict(x_test)


# Evaluate Model Accuracy
mod_acc = accuracy_score(y_test, pred)

# Return model metrics
st.markdown(f"<h5 style='color: white;'><b>Accuracy of {mod_sel} Model :</b> <i>{mod_acc:.2f}</i></h5>", unsafe_allow_html=True)

# if st.button('Random Forest Classifier Model'):
#     st.markdown(f"<h5 style='color: white;'><b>Accuracy of Random Forest Classifier Model :</b> <i>{mod_acc:.2f}</i></h5>", unsafe_allow_html=True)


# Custom data
# cust_pred = pipe.predict(X_test)
# cust_pred_list = list(cust_pred)


# for index in range(len(X_test)):
#     # insert predicted values to list
#     cust_pred_list.append(cust_pred)
    
#     # remove unneccessary value at last
#     cust_pred_list.pop()


# # Update Prediction Result
# predDf = testDf.copy()

# predDf['Heart_Dis_Pred'] = cust_pred_list


# Heading
st.markdown("<h2 style='color: white;'><b>Prediction of Heart Disease Detection</b></h2>", unsafe_allow_html=True)

# upd1, upd2 = st.columns([0.3, 0.7])
# with upd1:
# upload test dataset
uploaded_file = st.file_uploader("***Import file***", type=["csv"])
if uploaded_file is not None:
    upd_data = pd.read_csv(uploaded_file)
    
    cust_pred = pipe.predict(upd_data)
    cust_pred_list = list(cust_pred)


    for index in range(len(upd_data)):
        # insert predicted values to list
        cust_pred_list.append(cust_pred)
        
        # remove unneccessary value at last
        cust_pred_list.pop()
        
    ## Update Prediction Result
    upd_data['Heart Disease'] = cust_pred_list


    if st.button('Predict'):
        
        ## Update Prediction Result
        # predDf = test_data_pred.copy()
        # upd_data['Heart Disease'] = cust_pred_list


        ## Feature Engineering
        # One-Hot-Encoding for Resting blood pressure
        encoder = OneHotEncoder(categories='auto')
        encoded_data = encoder.fit_transform(upd_data[['Resting BPs']])

        # rbp_type = ['Normal', 'Elevated', 'Stage 1 hypertension', 'Stage1_Hypertension', 'Hypertensive_Crisis']
        # Create new dataframe with one-hot encoded features
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=[f'RBP_{i+1}' for i in range(encoded_data.shape[1])])

        # Concatenate encoded dataframe with original dataframe
        final_df = pd.concat([upd_data, encoded_df], axis=1)
        
        # Fetch columns from dataframe as list
        columns = final_df.columns.tolist()

        # Move the 3rd column to the last position
        columns.append(columns.pop(11))

        # Reorder the DataFrame columns
        final_df = final_df[columns]
        
        
        ## DataType Validation
        final_df['Gender'] = final_df['Gender'].astype(str)
        final_df['Heart Disease'] = final_df['Heart Disease'].astype(str)
        # final_df['Resting BPs'] = final_df['Resting BPs'].astype(str)
        final_df['Chest Pain Type'] = final_df['Chest Pain Type'].astype(str)
        final_df['Fasting Blood Sugar'] = final_df['Fasting Blood Sugar'].astype(str)
        final_df['Exercise Angina'] = final_df['Exercise Angina'].astype(str)

        ### Data Transformation
        # Categorize Gender
        def gender_class(gen):
            if gen == '1':
                return 'Male'
            elif gen == '0':
                return 'Female'
            else:
                return gen
        final_df['Gender'] = final_df['Gender'].apply(gender_class)

        # Categorize Fasting Exercise Angina
        def angina_class(angina):
            if angina == '1':
                return 'Yes'
            elif angina == '0':
                return 'No'
            else:
                return angina
        final_df['Exercise Angina'] = final_df['Exercise Angina'].apply(angina_class)

        # Categorize Fasting Blood Sugar
        def sugar_class(sugar):
            if sugar == '1':
                return 'Yes'
            elif sugar == '0':
                return 'No'
            else:
                return sugar
        final_df['Fasting Blood Sugar'] = final_df['Fasting Blood Sugar'].apply(sugar_class)


        # # Categorize Blood Pressure Type
        # def bp_class(bp):
        #     if bp < 120:
        #         return 1
        #     elif bp >= 120 & bp <=129:
        #         return 2
        #     elif bp >= 130 & bp <=139:
        #         return 3
        #     elif bp >= 140 & bp <=179:
        #         return 4
        #     elif bp >= 180:
        #         return 5
        #     else:
        #         return bp
        # final_df['Resting BPs'] = final_df['Resting BPs'].apply(bp_class)
        
        # Categorize Blood Pressure Type
        def bp_class(bp):
            if bp < 120:
                return 'Normal'
            elif bp >= 120 & bp <=129:
                return 'Elevated'
            elif bp >= 130 & bp <=139:
                return 'Stage-1 Hypertension'
            elif bp >= 140 & bp <=179:
                return 'Stage-2 Hypertension'
            elif bp >= 180:
                return 'Hypertensive Crisis'
            else:
                return bp
        final_df['Resting BPs'] = final_df['Resting BPs'].apply(bp_class)
        

        # Categorize Chest Pain Type
        def chest_class(chst):
            if chst == '1':
                return 'Typical Angina'
            elif chst == '2':
                return 'Atypical Angina'
            elif chst == '3':
                return 'Non-cardiac'
            elif chst == '4':
                return 'Stable Angina'
            else:
                return chst
        final_df['Chest Pain Type'] = final_df['Chest Pain Type'].apply(chest_class)

        # Categorize heart disease
        def heartdis_class(dis):
            if dis == '1':
                return 'Yes'
            elif dis == '0':
                return 'No'
            else:
                return dis
        # final_df['Heart Disease'] = final_df['Heart Disease'].apply(heartdis_class)
        final_df['Heart Disease'] = final_df['Heart Disease'].apply(heartdis_class)

        # Categorize Age
        def age_class(age):
                    if age >= 0 and age <= 30:
                        return '0-30'
                    elif age >= 31 and age <= 40:
                        return '31-40'
                    elif age >=41 and age <= 50:
                        return '41-50'
                    elif age >= 51 and age <= 60:
                        return '51-60'
                    elif age >=61 and age <= 70:
                        return '61-70'
                    elif age >=71 :
                        return '71-80'
                    else:
                        return age
                
        final_df['Age'] = final_df['Age'].apply(age_class)

        
        # Return Predicted Outcome
        # st.dataframe(predDf)
        st.dataframe(final_df)
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(final_df)

        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name='predicted_data.csv',
            mime='text/csv',
        )
        
        
        ## Feature Importance
        
        # feature selection
        if mod_sel == 'Decision Trees Classifier':
            # Access the model from the pipeline
            model = pipe.named_steps['Decision_tree'] 
        elif mod_sel == 'Random Forest Classifier':
            # Access the model from the pipeline
            model = pipe.named_steps['randomForest_tree']  
        elif mod_sel == 'Gradient Boosting Classifier':
            # Access the model from the pipeline
            model = pipe.named_steps['gradient_boosting'] 

        # Get feature importance
        feature_importance = model.feature_importances_
        feat_imp = [(feature_importance[i], final_df.columns[i]) for i in range(len(feature_importance))]
        df_fi = pd.DataFrame(feat_imp, columns=['Score', 'Columns'])
        df_fi.sort_values(by='Score', inplace=True, ascending=False)
        df_fi = round(df_fi,2)
        
        ## Visualization for Feature Importance    

        st.write('')
        st.write('')
        
        # Heading
        st.markdown("<h2 style='color: white;'><b>Feature Importance</b></h2>", unsafe_allow_html=True)
        
        # page layout
        col_fi1, col_fi2 = st.columns([0.8, 0.2])
        
        with col_fi1:
            fig = px.bar(df_fi, x='Score', y='Columns', color='Columns',
                        labels={'Score': 'Scores', 'Columns': 'Features'},
                        text='Score')

            # Set transparent background
            fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
            # Customize tick and label colors for x-axis and y-axis
            fig.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                            titlefont=dict(color='white'))  # Change x-axis label color to blue
            fig.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                            titlefont=dict(color='white'))  # Change y-axis label color to blue
                
            fig.update_layout(
                legend_title_text='Features',
                legend_title_font_color='white',
                legend_font_color='white',
                # legend_title_font_size=20,
                # legend_font_size=12
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
else:
    st.markdown("<p style='color: #d1e7dd;'><i>kindly upload the custom dataset for prediction</i></p>", unsafe_allow_html=True)
