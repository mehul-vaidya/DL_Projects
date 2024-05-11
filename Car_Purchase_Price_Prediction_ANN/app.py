"""
#conda create -p venv python==3.9
#conda activate [venv]
#pip install -r requirements.txt
#streamlit run app.py
"""
import os
_='''
2024-05-11 07:44:10.569217: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. 
You may see slightly different numerical results due to floating-point round-off errors from
 different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
'''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

ann = keras.models.load_model('Car_Purchase_Price_ANN.keras', compile=True)

_='''
input_array = [[50, 50000.500, 8650,444000]]
input_array_np = np.array(input_array)
print(input_array_np)
print(type(input_array_np))
print(ann.predict(input_array_np)[0][0])
'''


def car_purchasing_price_prediction(input_data):
    input_array_np = np.asarray(input_data)
    return ann.predict(input_array_np)[0][0]
  
def main():
    
    st.title('Car Price Prediction Web App')  

    #take input from user
    with st.form("Form 1",clear_on_submit=True):
      person_age = st.text_input('person age')
      annual_salary = st.text_input('annual salary')
      credit_card_debt = st.text_input('credit card debt')
      person_net_worth = st.text_input('person net worth')

      s_state=st.form_submit_button('Predict Car Purchasing Price')
      if s_state:
        if(person_age.isdigit() and annual_salary.isdigit() and credit_card_debt.isdigit() and person_net_worth.isdigit() and int(person_age)>18 and int(person_age)<100 ):
          person_age=float(person_age)
          annual_salary=float(annual_salary)
          credit_card_debt=float(credit_card_debt)
          person_net_worth=float(person_net_worth)

          prediction = car_purchasing_price_prediction([[person_age,annual_salary,credit_card_debt,person_net_worth]])
          st.success("Customer predicted budget is " + str('{0:.2f}'.format(prediction)) + " USD")  
                    
        else:
          st.error("You have entered incorrect data")  

if __name__ == '__main__':
    main()
    

