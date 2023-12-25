import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")

# Load data
@st.cache
def load_data(data):
    df=pd.read_csv(data,index_col=False)
    return df

def run_eda_app():
    st.title("Exploratory Data Analysis")
    df=load_data("./Data/df_1.csv")
    # df_new=load_data("./parkinson_upsample.csv")
    #df_encoded=load_data("./data/diabetes_data_upload_clean.csv") 
    #freq_df=load_data("./data/freqdist_of_age_data.csv")
    
    submenu=st.sidebar.selectbox("Submenu",["Descriptive","Plots"])
    if submenu=="Descriptive":
        st.subheader("Descriptive Data")
        st.dataframe(df)
        with st.expander("Data Types"):
            st.dataframe(df.dtypes.astype(str))
        with st.expander("Descriptive Summary"):
            st.dataframe(df.describe().astype(str))
        
        
