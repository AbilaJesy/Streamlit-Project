import streamlit as st
import pandas as pd
import os 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Salary Predicter")
st.image("media//img1.jpg",width=800)


data=pd.read_csv("data//Salary_Data.csv")
x = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))
# st.sidebar.markdown("<i><h1> LOGO</h1><i>",True)

selected_option = st.sidebar.radio("# LOGO", ["Home", "Prediction", "Contribute"])

# You can use the selected_option variable in your application logic
if selected_option == "Home":

    if st.checkbox("Show Tables"):
        st.table(data)

    graph=st.selectbox("What kind of graph you went?...",["Non-Interactive","Interactive"])

    val = st.slider("Filter data using years",0,10)
    data = data.loc[data["YearsExperience"]>= val]

    if graph == "Non-Interactive":
        fig, ax = plt.subplots()
        plt.figure(figsize = (10,5))
        ax.scatter(data["YearsExperience"],data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot(fig)
    if graph == "Interactive":
        layout =go.Layout(
            xaxis = dict(range=[0,16]),
            yaxis = dict(range =[0,210000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'),layout = layout)
        st.plotly_chart(fig)


elif selected_option == "Prediction":
    st.header("Know your Salary")
    val = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    val = np.array(val).reshape(1,-1)
    pred =lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")

elif selected_option == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)
    if st.button("submit"):
        to_add = {"YearsExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//Salary_Data.csv",mode='a',header = False,index= False)
        st.success("Submitted")
