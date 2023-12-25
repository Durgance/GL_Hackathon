import streamlit as st

from eda_app import run_eda_app
from ml_app import run_ml_app

st.set_page_config(page_title="House Price Regression - GL Hackathon",
                        page_icon="🦍")




def main():
    menu=["Home","EDA","ML Model","About"]
    choice=st.sidebar.selectbox("Menu", menu)
    if choice=="Home":
        st.title("House Price Regression - GL Hackathon")
        st.image("78036515.cms")
        st.write("""
			
                Housing price prediction
                Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to a north-south railroad. 
                 
                House price negotiations often have a lot of influencing factors and not just the number of bedrooms or the position of the kitchen.Take the given dataset with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. In this hackathon, predict the final price of each home. The application should be modeled using Machine Learning, you may explore libraries such as PySpark. Apply containerization principles as a better software engineering practice. You may explore Kafka server for streaming the data.

                The model can be deployed using Docker containers for scalability.

                Dataset: "https://www.kaggle.com/c/house-prices-advanced-regression-techniques/"
			
                App Content:
                 
                    - EDA Section: Exploratory Data Analysis of Data
                    - ML Section: ML Predictor App

			""")
        pass
    elif choice=="EDA":
        run_eda_app()

        pass
    elif choice=="ML Model":
        run_ml_app()
        pass
    else:
        st.title("About")
        st.subheader("Durgance Gaur")
        # st.subheader("Uber")

        st.markdown("""
        * ### Description :

            * ##### The dataset was in the form of csv file.
            * ##### The challenge required to understand the problem statement effectively.
            * ##### Working on data preprocessing, data analysis and working with NULL values.
            * ##### Modeling Data to get in PySpark using various modeling techniques

        * ### Metadata :

            * ##### The dataset was in the form of csv file .
            """)

    
if __name__=="__main__":
    main()