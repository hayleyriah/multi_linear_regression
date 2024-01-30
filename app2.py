import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#load the dataset

df = pd.read_excel("Dropoutdataset.xlsx")

#Add logo

st.image("Africdsa.jpeg")

#add title to app

st.title("Multiple Regression App")

#Add the header

st.header("Dataset Concept.", divider="rainbow")

#Add paragraph explaining the dataset

st.write("""The Dropout dataset is a comprehensive collection of information related to students' academic performance and various socio-economic factors, 
            aimed at understanding the factors influencing students decisions to either graduate, dropout, or remain enrolled in educational institutions.
            This dataset includes features such as socio-economic background, parental education, academic scores, attendance,and extracurricular activities.
            In the context of multi-linear regression, researchers and 
            data scientists utilize this dataset to build predictive models that can assess the likelihood of a student either graduating, 
            dropping out, or remaining enrolled based on a combination of these factors. By employing multi-linear regression techniques, 
            the dataset allows for the examination of the relationships and interactions among multiple independent variables simultaneously. 
            The model seeks to identify which specific factors play a significant role in predicting the educational outcomes of students, 
            providing valuable insights for educators, policymakers, and institutions to implement targeted interventions and support systems for at-risk students. 
            Through the analysis of the Dropout dataset, it becomes possible to develop more informed strategies to improve overall student success and reduce dropout rates.""")


#-------------------------------------------------------DISPLAY OR EDA----------------------------------------------

st.header("Exploratory Data Analysis (EDA)", divider="rainbow")


if st.checkbox("Dataset info"):
     st.write("Dataset info", df.info())
     
if st.checkbox("Number of Rows"):
     st.write("Number of Rows", df.shape[0])
     
if st.checkbox("Number of Columns"):
     st.write("Number of Columns", df.columns.tolist())
     
if st.checkbox("Data types"):
     st.write("Data types", df.dtypes)
     
if st.checkbox("Missing Values"):
     st.write("Missing Values", df.isnull().sum())
     
if st.checkbox("Statistical Summary"):
     st.write("Statistical Summary", df.describe())
     
     
#==============================================Visualization===============================

st.header("VIsualization of the Dataset (VIZ)", divider="rainbow")


#bar chart

if st.checkbox("Inflation Against GDP Bar Chart"):
    st.write("Bar Chart of Inflation rate Against GDP")
    st.bar_chart(x = "Inflation rate" , y="GDP" , data=df , color=["#FF0000"])
    

#CREATE THE BAR CHART

if st.checkbox("Gender Bar Chart"):
    st.write("Bar Chart for Gender rate Against GDP")
    st.bar_chart(x = "Gender" , y="GDP" , data=df , color=["#FF0000"])
        

#create line chart

if st.checkbox("Inflation Rate Line Chart"):
    st.write("Line Chart for Inflation rate Against GDP")
    st.bar_chart(x = "Inflation rate" , y="GDP" , data=df , color=["#ffaa0088"])
    
    
#create scatter plot

if st.checkbox("Scatter Plot"):
    st.write("Scatter Chart of GDP Against Target")

#create the histogram using Altair
st.scatter_chart(
    x="Target",
    y='GDP',
    data = df,
    color=["#ffaa0088"]
)



#-------------------------Multiple Linear Regression Model---------------------------------

#Enconding the target column using labelencoder

university = LabelEncoder()
df['Target'] = university.fit_transform(df['Target'])

#Use onehotencoder to encode the categorical features

ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(),['Target'])], remainder='passthrough')  
X = df.iloc[:,: -1] 
y = df.iloc[:,-1] 
y_encoded = ct.fit_transform(df[["Target"]])

#splitting data into training and testing

X_train ,X_test, y_train, Y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=0)

#fit the regresion model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#User input for independent variables
st.sidebar.header ("Enter values to be Predicted", divider='rainbow')

#create the input for each feature

user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}", 0.0)                        
    

#Button to trigger the prediction

if st.sidebar.button("Predict"):
     
     #create a dataframe for the user input
     user_input_df = pd.DtatFrame([user_input],dtype = float)
     
     #predict using the trained model
     y_pred = regressor.predict(user_input)
     
     #inverse transform to get the original target values
     
     predicted_class = university.inverse_transform(np.array(y_pred, axis=1))
     
     #display the predicted class
     st.header("Predicted Result Outcome:", divider='rainbow')
     st.write(predicted_class[0])