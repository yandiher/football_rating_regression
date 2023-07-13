# ==========
# data section

import pandas as pd
pd.set_option('display.max_columns', None)

data = pd.read_csv('soccer_player_2020.csv')
data_select = data[['overall','pace','shooting','passing','dribbling','defending','physic']]

data_select = data_select.dropna()

y = data_select.overall
x = data_select.drop(['overall'], axis=1)

# ==========
# machine learning section

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# from sklearn.ensemble import ExtraTreesRegressor
# extra = ExtraTreesRegressor()
# extra.fit(x_train, y_train)

import pickle

clf = pickle.load(open('clf.pkl', 'rb'))

# ==========
# streamlit section

def regression(speed, shooting, passing, dribbling, defending, physic):
    input_number = [[speed, shooting, passing, dribbling, defending, physic]]
    input_number = scaler.transform(input_number)
    prediksi_web = clf.predict(input_number)
    return prediksi_web

import streamlit as st

def run():
    st.title('Prediksi Nilai Pemain Sepak Bola')

    speed = st.slider('speed', 1, 99, 50)
    shooting = st.slider('shooting', 1, 99, 50)
    passing = st.slider('passing', 1, 99, 50)
    dribbling = st.slider('dribbling', 1, 99, 50)
    defending = st.slider('defending', 1, 99, 50)
    physic = st.slider('physic', 1, 99, 50)

    prediction=""
    if st.button("submit"):
        prediction = int(regression(speed, shooting, passing, dribbling, defending, physic))

    st.success(f'skor untuk pemain tersebut adalah : {prediction}')

if __name__ == '__main__':
    run()