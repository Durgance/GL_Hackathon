import streamlit as st 

import os
import pickle
import pandas
from pyspark.ml.tuning import CrossValidatorModel
import numpy as np
import pyspark
attrib_info="""
#### Attribute Information : 
    - We have 23 feature in the dataset
    - 1 Target Feature 
"""
import pandas as pd
import numpy as np
import os
# os.environ["JAVA_HOME"]="/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"]="./spark-3.3.1-bin-hadoop3"
from pyspark.sql import SparkSession
import findspark
from pyspark import SparkConf
from pyspark import SparkContext

findspark.init()
findspark.find()
# spark = SparkSession.builder.appName('House_Price_Regression').getOrCreate()


def _initialize_spark():
    """Create a Spark Context for Streamlit app"""
    conf = SparkConf().setAppName("House_Price_Regression").setMaster("local")
    sc = SparkContext.getOrCreate(conf=conf)
    return sc

df = pd.read_csv("./Data/df_1.csv")
df = df.iloc[: , 1:]
categorical_cols = []
numerical_cols = []
label_col = ["SalePrice"]
# Alley, FireplaceQu, PoolQC, Fence, MiscFeature have too much null
# Id has too many categories
# Utilities, Condition2, RoofMatl, Heating has one weak categories and binary, shoud be discarted
discarted_cols = ["Id", "Alley","FireplaceQu","PoolQC","Fence","MiscFeature","Utilities","Condition2","RoofMatl","Heating"]

categorical_cols = []
numerical_cols = []

def grab_cat_num_cols(dataframe):
    for column in df.columns:
        if (column not in label_col+discarted_cols):
            if df[column].dtype == 'object':
                categorical_cols.append(column)
            else:
                numerical_cols.append(column)

    return categorical_cols, numerical_cols

categorical_cols, numerical_cols = grab_cat_num_cols(df)

# @st.cache
def load_model():
    # with open(filename,"rb") as f:
    #     model=pickle.load(f)
    model = CrossValidatorModel.load("./GBT_Reg_CV")
    return model
model=load_model()
def run_ml_app():
    st.title("Predicting Price of your House")
    
    #with 
    spark = _initialize_spark()


    with st.expander("Attribute Info"):
        st.markdown(attrib_info)

    col1,col2=st.columns(2)

    with col1:
        cat_col_input ={}
        for col in categorical_cols:
            cat_col_input[f"{col}"] = st.selectbox(f"{col}:",list(df[col].unique()))
        

    with col2:
        num_col_input ={}
        for col in numerical_cols:
            num_col_input[f"{col}"] = st.slider(f"{col}:",df[col].min(),df[col].max(),5)
    
        
    with st.expander("Your Selected Options are "):
        results={}
        results.update(cat_col_input)
        results.update(num_col_input)
        st.write(results)

        encoded_result = pd.DataFrame(results,columns = results.keys(),index=[0])    
        from pyspark.sql import SQLContext

        sqlContext = SQLContext(spark)
        encoded_result = sqlContext.createDataFrame(encoded_result)
        # encoded_result = spark.createDataFrame(encoded_result)
        # encoded_result= []
        # for i in results.values(): 
        #     # if type(i)==float:
        #     encoded_result.append(i)
        
            

    st.write("""
    # WAITING FOR PREDICTION !!!! 
""")
    #st.write(np.array(encoded_result).reshape(1,-1))
    with st.expander("Predicting Price of your House"):
        
        # rf_scaler=load_model("./RandomForest/scaler_1.pkl")
        
        # single_samlpe=np.array(encoded_result).reshape(1,-1)
        single_sample = encoded_result
        #st.write()
        #st.write(rf_scaler.transform(single_samlpe))

        prediction=model.transform(single_sample)
        # pred_prob=model.predict_proba(single_samlpe)
        #st.write(prediction)
        #st.write(pred_prob)
        result_price = prediction.select('prediction')
        st.write(f"Pirce of Home : {round(float(result_price.first()[0]),2)}")
        # if prediction==1:
        #     st.warning(f"High Risk of Parkinson's 💀")
        #     predict_probability_score={"Negative Risk":pred_prob[0][0]*100,
        #     "Postive Risk":pred_prob[0][1]*100}
        #     st.write(predict_probability_score)
        # else:
        #     st.success(f"You are Healthy 💘")
        #     predict_probability_score={"Negative Risk":pred_prob[0][0]*100,
        #     "Postive Risk":pred_prob[0][1]*100}
        #     st.write(predict_probability_score)