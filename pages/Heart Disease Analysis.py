import pandas as pd
import streamlit as st 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

st.set_page_config(layout="wide",page_title="Heart Disease Analysis")


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


# Fetch data
data = pd.read_csv('/mount/src/heart-disease-detection/heart_dis.csv')
df = pd.DataFrame(data)


### Data Preprocessing

## DataType Validation
df['Gender'] = df['Gender'].astype(str)
df['Heart Disease'] = df['Heart Disease'].astype(str)
df['Chest Pain Type'] = df['Chest Pain Type'].astype(str)
df['Fasting Blood Sugar'] = df['Fasting Blood Sugar'].astype(str)
df['Exercise Angina'] = df['Exercise Angina'].astype(str)

### Data Transformation

# Categorize Gender
def gender_class(gen):
    if gen == '1':
        return 'Male'
    elif gen == '0':
        return 'Female'
    else:
        return gen
df['Gender'] = df['Gender'].apply(gender_class)

# Categorize Fasting Exercise Angina
def angina_class(angina):
    if angina == '1':
        return 'Yes'
    elif angina == '0':
        return 'No'
    else:
        return angina
df['Exercise Angina'] = df['Exercise Angina'].apply(angina_class)

# Categorize Fasting Blood Sugar
def sugar_class(sugar):
    if sugar == '1':
        return 'Yes'
    elif sugar == '0':
        return 'No'
    else:
        return sugar
df['Fasting Blood Sugar'] = df['Fasting Blood Sugar'].apply(sugar_class)


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
df['Chest Pain Type'] = df['Chest Pain Type'].apply(chest_class)

# Categorize heart disease
def heartdis_class(dis):
    if dis == '1':
        return 'Yes'
    elif dis == '0':
        return 'No'
    else:
        return dis
df['Heart Disease'] = df['Heart Disease'].apply(heartdis_class)

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
        
df['Age'] = df['Age'].apply(age_class)

# st.write(df)


head_1, head_2, head_3 = st.columns([20,60, 20])
        
with head_2:
    st.markdown("<h1 style='color:white;text-align:center;'>Cardio Analysis for Heart Disease Detection</h1>", unsafe_allow_html=True)
    
# For Blank Space
blank1, blank2 = st.columns(2)
blank3, blank4 = st.columns(2)
blank4, blank6 = st.columns(2)


### Visualizations

# page layout
rw1a, rw1b, rw1d, rw1d, rw1e = st.columns([5, 42.5, 5, 42.5, 5])

with rw1b:
    # Heading
    # st.write('##### Analysis of Heart Disease Distribution')
    st.markdown("<h4 style='color:white;'><b>Analysis of Heart Disease Distribution</b></h4>", unsafe_allow_html=True)
    
    cnt_heart_dis = df.groupby('Heart Disease').size().reset_index(name='Count')
    
    fig = px.pie(cnt_heart_dis, values='Count', names='Heart Disease', hole=0.5)
    
    # Set transparent background
    fig.update_layout(
                    # legend=dict(title=dict(text='Heart Disease')),
                      plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    fig.update_layout(
    legend_title_text='Heart Disease',
    legend_title_font_color='white',
    legend_font_color='white',
    # legend_title_font_size=20,
    # legend_font_size=12
    )
    
    # Customize tick and label colors for x-axis and y-axis
    fig.update_xaxes(tickfont=dict(color='white'), 
                titlefont=dict(color='white'))  
    fig.update_yaxes(tickfont=dict(color='white'), 
                titlefont=dict(color='white')) 
        
    st.plotly_chart(fig, use_container_width=True)
    
    
    
with rw1d:
    # Heading
    # st.write('##### Patient Distribution by Chest Pain Type')
    st.markdown("<h4 style='color:white;'><b>Patient Distribution by Chest Pain Type</b></h4>", unsafe_allow_html=True)
        
    cnt_chst_pain = df.groupby('Chest Pain Type').size().reset_index(name='Count')
    
    fig = px.bar(cnt_chst_pain, x='Chest Pain Type', y='Count', color='Chest Pain Type',
                labels={'Count': 'Count of Patients'},
                text='Count')

    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    fig.update_layout(
    legend_title_text='Chest Pain Type',
    legend_title_font_color='white',
    legend_font_color='white',
    # legend_title_font_size=20,
    # legend_font_size=12
    )
    
    # Customize tick and label colors for x-axis and y-axis
    fig.update_xaxes(tickfont=dict(color='white'), 
                    titlefont=dict(color='white')) 
    fig.update_yaxes(tickfont=dict(color='white'),  
                        titlefont=dict(color='white')) 
    # Render the chart using Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    
# page layout
rw2a, rw2b, rw2d, rw2d, rw2e = st.columns([5, 42.5, 5, 42.5, 5])

with rw2b:
    # Heading
        # st.write('##### Distribution of Volume of Shale (VSH)')
        st.markdown("<h4 style='color:white;'><b>Gender-wise Variation in Heart Disease</b></h4>", unsafe_allow_html=True)
        cnt_dis_gen_m = df[df['Gender'] == 'Male'].groupby(['Heart Disease']).size()
        cnt_dis_gen_f = df[df['Gender'] == 'Female'].groupby(['Heart Disease']).size()
        
        cnt_dis_gen_df = pd.DataFrame({
                        'Heart Disease': cnt_dis_gen_m.index,
                        'Male': cnt_dis_gen_m.values,
                        'Female': cnt_dis_gen_f.values
                    })
        # st.write(cnt_cigs_gen_df)
        cnt_dis_gen_df.sort_values(by='Heart Disease', inplace=True, ascending=False)
        fig = px.bar(barmode='group')
                
        # Add traces for Female and Male separately
        fig.add_bar(y=cnt_dis_gen_df['Heart Disease'], x=cnt_dis_gen_df['Male'], name='Male', marker_color='#0068c9', text=cnt_dis_gen_df['Male'], orientation='h')
        fig.add_bar(y=cnt_dis_gen_df['Heart Disease'], x=cnt_dis_gen_df['Female'], name='Female', marker_color='#83c9ff', text=cnt_dis_gen_df['Female'], orientation='h')
        
        # Update layout to show dual y-axes
        fig.update_layout(yaxis=dict(title='Heart Disease', side='left', showgrid=False),
                        yaxis2=dict(overlaying='y', side='right', showgrid=False),
                        legend=dict(title=dict(text='Gender'), x=1, y=1.2))
        fig.update_xaxes(title='Count of Patients')
        
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
        fig.update_layout(
        legend_title_text='Gender',
        legend_title_font_color='white',
        legend_font_color='white',
        # legend_title_font_size=20,
        # legend_font_size=12
        )
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                    titlefont=dict(color='white'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                    titlefont=dict(color='white'))  # Change y-axis label color to blue
            
        st.plotly_chart(fig, use_container_width=True)
        
with rw2d:
    # Heading
        # st.write('##### Distribution of Volume of Shale (VSH)')
        st.markdown("<h4 style='color:white;'><b>Distribution of Heart Rate (BPM)</b></h4>", unsafe_allow_html=True)
        
        # fig1 = px.violin(df, y='VSH')
        fig1 = px.box(df, y='Max Heart Rate', color_discrete_sequence=["#83c9ff"])

        fig1.update_traces(boxmean=True, jitter=0.5, whiskerwidth=0.2)
        
        # Set transparent background
        fig1.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig1.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                        titlefont=dict(color='white'))  # Change x-axis label color to blue
        fig1.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                        titlefont=dict(color='white'))  # Change y-axis label color to blue
        
        st.plotly_chart(fig1, use_container_width=True)
        
     
# page layout
rw3a, rw3b, rw3d, rw3d, rw3e = st.columns([5, 42.5, 5, 42.5, 5])

with rw3b:
    # Heading
        # st.write('##### Distribution of Volume of Shale (VSH)')
        st.markdown("<h4 style='color:white;'><b>Distribution of Heart Disease By Age</b></h4>", unsafe_allow_html=True)   
        
        counts_age_hd = df.groupby(['Heart Disease', 'Age']).size().reset_index(name='Count')
        counts_age_hd.sort_values(by='Age', ascending=True)
        # st.write(counts_age_hd)

        fig = px.sunburst(counts_age_hd, path=['Age','Heart Disease'], values='Count')
        
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                        titlefont=dict(color='white'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                        titlefont=dict(color='white'))  # Change y-axis label color to blue
        
        st.plotly_chart(fig, use_container_width=True)
        
    
with rw3d:
    # Heading
        st.markdown("<h4 style='color:white;'><b>Analysis of Average Cholesterol Across Age</b></h4>", unsafe_allow_html=True)   

        avg_age_chol = df.groupby('Age')['Cholesterol'].mean().round(2).reset_index(name='Average')
        # avg_age_chol = avg_age_chol.sort_values(by='Age')
        fig = px.line(avg_age_chol, x='Age', y='Average',
                    labels={'Average': f'Average of Cholesterol'},
                    text=f'Average')
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                        titlefont=dict(color='white'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                        titlefont=dict(color='white'))  # Change y-axis label color to blue        
        
        fig.update_traces(
                        textposition='top center',  # Move text to the top center of each point
                        textfont=dict(color='white'),  # Customize text font
                        texttemplate='%{text:.1f}',  # Format text
                        hoverinfo='skip',   # Hide hover info to only display text
                        )  

        st.plotly_chart(fig, use_container_width=True)
        
        
# page layout
rw4a, rw4b, rw4d, rw4d, rw4e = st.columns([5, 42.5, 5, 42.5, 5])

with rw4b:
    # Heading
        st.markdown("<h4 style='color:white;'><b>Count of Patients with Blood Sugar > 120 mg/dl by Heart Disease</b></h4>", unsafe_allow_html=True)   
        
        grouped_df = df.groupby(['Fasting Blood Sugar', 'Heart Disease']).size().reset_index(name='Count')
        
        filter_option = ('Yes', 'No')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_db = st.selectbox("***Blood Sugar > 120mg/dl***", filter_option)
            
        # Filter for 'Fasting Blood Sugar' greater than 120 mg/dl
        filtered_df = grouped_df[grouped_df['Fasting Blood Sugar'] == selected_db]
        # st.write(filtered_df)
        
        # Create custom plotly chart
        fig = px.bar(filtered_df, x='Heart Disease', y='Count', color='Heart Disease', text='Count',
                    labels={'Count': 'Number of Patients', 'Heart Disease': 'Heart Disease'})
        fig.update_layout(
                        xaxis_title='Heart Disease',
                        yaxis_title='Number of Patients')
        
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                        titlefont=dict(color='white'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                        titlefont=dict(color='white'))  # Change y-axis label color to blue        

        fig.update_layout(
        legend_title_text='Heart Disease',
        legend_title_font_color='white',
        legend_font_color='white',
        # legend_title_font_size=20,
        # legend_font_size=12
        )
        
        st.plotly_chart(fig, use_container_width=True)
        

with rw4d:
    # Heading
        st.markdown("<h4 style='color:white;'><b>Heart Disease Caused by Angina Exercise</b></h4>", unsafe_allow_html=True)   
        
        
        filter_option = ('All', 'Yes', 'No')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_gen = st.selectbox("***Angina Exercise***", filter_option)
                    
        if selected_gen == 'All':
            filtered_df_ea = df[df['Exercise Angina'].isin(['Yes', 'No'])]
        elif selected_gen == 'Yes':
            filtered_df_ea = df[df['Exercise Angina'].isin(['Yes'])]
        elif selected_gen == 'No':
            filtered_df_ea = df[df['Exercise Angina'].isin(['No'])]
            
        filtered_df_ea1 = filtered_df_ea[filtered_df_ea['Heart Disease'].isin(['Yes'])]
        
        cnt_ea_gen_m = filtered_df_ea1[filtered_df_ea1['Gender'] == 'Male'].groupby(['Exercise Angina']).size()
        cnt_ea_gen_f = filtered_df_ea1[filtered_df_ea1['Gender'] == 'Female'].groupby(['Exercise Angina']).size()

        cnt_ea_gen_df = pd.DataFrame({
                        'Exercise Angina': cnt_ea_gen_m.index,
                        'Male': cnt_ea_gen_m.values,
                        'Female': cnt_ea_gen_f.values
                    })
            
        fig = px.bar(barmode='group')
                
        # Add traces for Female and Male separately
        fig.add_bar(x=cnt_ea_gen_df['Exercise Angina'], y=cnt_ea_gen_df['Male'], name='Male', marker_color='#0068c9', text=cnt_ea_gen_df['Male'])
        fig.add_bar(x=cnt_ea_gen_df['Exercise Angina'], y=cnt_ea_gen_df['Female'], name='Female', marker_color='#83c9ff', text=cnt_ea_gen_df['Female'])
        
        # Update layout to show dual y-axes
        fig.update_layout(yaxis=dict(title='Count of Patients', side='left', showgrid=False),
                        yaxis2=dict(overlaying='y', side='right', showgrid=False),
                        # legend=dict(
                        #     title=dict(text='Gender'),
                        #             x=1, y=1.2)
                        )
        fig.update_xaxes(title='Angina Exercise')
        
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='white'),  # Change x-axis tick color to blue
                        titlefont=dict(color='white'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='white'),  # Change y-axis tick color to green
                        titlefont=dict(color='white'))  # Change y-axis label color to blue
            
        fig.update_layout(
            legend_title_text='Gender',
            legend_title_font_color='white',
            legend_font_color='white',
            # legend_title_font_size=20,
            # legend_font_size=12
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
