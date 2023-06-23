# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:42:54 2023

@author: Tushar
"""

import pandas as pd
import pickle
import streamlit as st


st.title('Customer Personality Analysis')

st.sidebar.header('User Input Parameters')

load = open('model.pkl','rb')
model = pickle.load(load)


def user_input_features():
    
    Year_Birth = st.sidebar.number_input('Year_Birth', min_value=0, max_value=2100)
    Education = st.sidebar.selectbox('Education', ('Graduation','PhD','Master','Basic','2n Cycle'))
    Marital_Status = st.sidebar.selectbox('Marital_Status', ('Single','Together','Married','Divorced','Widow','Alone','Absurd','YOLO'))
    Income = st.sidebar.number_input('Income', min_value=0, max_value=1000000)
    Kidhome = st.sidebar.slider('Kidhome', min_value=0, max_value=5)
    Teenhome = st.sidebar.slider('Teenhome', min_value=0, max_value=5)
    Recency = st.sidebar.number_input('Recency', min_value=0, max_value=365)
    MntWines = st.sidebar.number_input('MntWines', min_value=0, max_value=1000)
    MntFruits = st.sidebar.number_input('MntFruits', min_value=0, max_value=1000)
    MntMeatProducts = st.sidebar.number_input('MntMeatProducts', min_value=0, max_value=1000)
    MntFishProducts = st.sidebar.number_input('MntFishProducts', min_value=0, max_value=1000)
    MntSweetProducts = st.sidebar.number_input('MntSweetProducts', min_value=0, max_value=1000)
    MntGoldProds = st.sidebar.number_input('MntGoldProds', min_value=0, max_value=1000)
    NumDealsPurchases = st.sidebar.number_input('NumDealsPurchases', min_value=0, max_value=100)
    AcceptedCmp1 = st.sidebar.selectbox('AcceptedCmp1',('0','1'))
    AcceptedCmp2 = st.sidebar.selectbox('AcceptedCmp2',('0','1'))
    AcceptedCmp3 = st.sidebar.selectbox('AcceptedCmp3',('0','1'))
    AcceptedCmp4 = st.sidebar.selectbox('AcceptedCmp4',('0','1'))
    AcceptedCmp5 = st.sidebar.selectbox('AcceptedCmp5',('0','1'))
    Response = st.sidebar.selectbox('Response',('0','1'))
    NumWebPurchases = st.sidebar.number_input('NumWebPurchases', min_value=0, max_value=100)
    NumCatalogPurchases = st.sidebar.number_input('NumCatalogPurchases', min_value=0, max_value=100)
    NumStorePurchases = st.sidebar.number_input('NumStorePurchases', min_value=0, max_value=100)
    NumWebVisitsMonth = st.sidebar.number_input('NumWebVisitsMonth', min_value=0, max_value=100)
        
    
    data = {'Year_Birth':Year_Birth,
            'Education':Education,
            'Marital_Status':Marital_Status,
            'Income':Income,
            'Kidhome':Kidhome,
            'Teenhome':Teenhome,
            'Recency':Recency,
            'MntWines':MntWines,
            'MntFruits':MntFruits,
            'MntMeatProducts':MntMeatProducts,
            'MntFishProducts':MntFishProducts,
            'MntSweetProducts':MntSweetProducts,
            'MntGoldProds':MntGoldProds,
            'NumDealsPurchases':NumDealsPurchases,
            'AcceptedCmp1':AcceptedCmp1,
            'AcceptedCmp2':AcceptedCmp2,
            'AcceptedCmp3':AcceptedCmp3,
            'AcceptedCmp4':AcceptedCmp4,
            'AcceptedCmp5':AcceptedCmp5,
            'Response':Response,
            'NumWebPurchases':NumWebPurchases,
            'NumCatalogPurchases':NumCatalogPurchases,
            'NumStorePurchases':NumStorePurchases,
            'NumWebVisitsMonth':NumWebVisitsMonth}
    
    
    features = pd.DataFrame(data,index = [0])
    return features
    
    
df = user_input_features()
st.subheader('User Input parameters')
st.markdown('Customer Information')
st.write(df)


st.subheader('Press below to get customer segment')
if st.button('Predict'):
    result = model.predict(df)
    st.success('Cluster Number : {} '.format(result))


    