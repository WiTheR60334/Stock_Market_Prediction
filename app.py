import streamlit as st
import datetime as dt
import streamlit as st
import numpy as np
import pandas as pd
import math, datetime
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from keras.layers import Dense,Dropout, LSTM
# from keras.models import Sequential
from keras.models import load_model


input = st.sidebar.text_input('**Enter stock ticker**', 'AAPL')
StockIsValid = yf.download(input)
if StockIsValid.empty:
    st.error('**Error : Invalid Stock Ticker**\n\n **Enter registered stock ticker name**')
else:
    today = dt.date.today()
    default_start_date=datetime.date(2012,1,1)
    min= dt.datetime(2010, 1, 1).date()
    st.sidebar.success('**There is difference of 700 days between start date and end date.**')
    end_date = st.sidebar.date_input('**End date**', today, min_value=min, max_value=today)  
    max = end_date - dt.timedelta(days=700)
    start_date = st.sidebar.date_input('**Start date**', default_start_date, min_value=min, max_value=max)

    if start_date < end_date:
        st.sidebar.success('**Start date:**`%s`\n\n**End date:**`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must be greater than start date.')
    
    st.title('Fundamental Analysis')
    
    two_days_ago = today - datetime.timedelta(days=2)
    yesterday = today - datetime.timedelta(days=1)
    df = yf.download(input, start='2012-01-01', end=today)
    df1 = yf.download(input, start=today, end=today)
    df2 = yf.download(input, start=two_days_ago, end=today)
    df3 = yf.download(input, start=start_date, end=end_date)
    close_value = df2['Close']
    
    st.subheader('Data From 2012-2023')
    st.write(df.describe())
    
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df3.Close)
    plt.grid(True)
    st.pyplot(fig)
        
    st.subheader('Closing Price vs Time Chart with 100 days MA')
    MA100 = df3.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(MA100)
    plt.plot(df3.Close.rolling(100).mean())
    plt.plot(df3.Close)
    st.pyplot(fig)
    
    st.subheader('Closing Price vs Time Chart with 200 days MA')
    
    moving_average_100 = sum(df3['Close'][-100:]) / 100
    moving_average_200 = sum(df3['Close'][-200:]) / 200
    
    if moving_average_100 > moving_average_200:
        trend="Bullish"
    else:
        trend="Bearish"   
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(f"<p style='font-size: 24px;'><b>Stock trend : {trend}</b></p>", unsafe_allow_html=True)     
    st.markdown("<br>", unsafe_allow_html=True)
    
    MA200 = df3.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(MA100, label='MA100')
    plt.plot(MA200, label='MA200')
    plt.plot(df3.Close, label='Actual Price')
    plt.legend()
    st.pyplot(fig)
    
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.8)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.8):int(len(df))])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    model = load_model('Stock_Prediction_keras_model.h5')
    
    data_training_array = scaler.fit_transform(data_training)
    
    X_train = []
    y_train = []
    
    for i in range(100, data_training_array.shape[0]):
        X_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    previous_100_days_data = data_training.tail(100)
    final_df = previous_100_days_data.append(data_testing, ignore_index=True)
    scale_testing = scaler.fit_transform(final_df)
    
    X_test = []
    y_test = [] 
    
    for i in range(100, scale_testing.shape[0]):
        X_test.append(scale_testing[i - 100:i])
        y_test.append(scale_testing[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_predicted = model.predict(X_test)
    
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Prediciton of Closing Price")
    last_100_days = df['Close'].tail(100).values
    last_100_days=np.array(last_100_days)
    min=np.min(last_100_days)
    max=np.max(last_100_days)
    last_100_days_scaled = (last_100_days.reshape(-1, 1) - min) / (max-min)
    X_test = []
    X_test.append(last_100_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    predicted_price = predicted_price * (max- min) + min
    # predicted_price = predicted_price[-1][-1]
    
    
    display_option = st.select_slider("**Two Algos:**", ["1st Algo", "2nd Algo"])
    
    if display_option == "2nd Algo":
        st.subheader('Actual Price Vs Predicted Price')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Actual Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        if close_value.empty:
            st.write(f"**{two_days_ago} Closing price was : Stock Market was closed!!!**")
        else :
            st.write(f'**{two_days_ago} Closing price was :  {close_value.iloc[0]}**')
        try:
            if df1.empty:
                st.write(f"**{yesterday} Closing price was : Stock Market was closed!!!**")
            else:
                todays_close_price = df1['Close'].iloc[0]
                st.write(f'**{yesterday} Closing price was :  {todays_close_price}**')
        except FileNotFoundError:
            st.write("**{yesterday} Closing price was : Stock Market was closed !!!**")
        st.write(f'Todays Closing price might be : {predicted_price[0][0]}')
    
        st.subheader('Closing Data')
        a=df['Close'].tail(10)
        st.dataframe(a)
    
    else:
        #Different Algo
        data=df.filter(['Close'])
        dataset=data.values
        data_training_len=math.ceil(len(dataset)*0.8)
        
        scaler = MinMaxScaler(feature_range=(0,1))
        data_training_array=scaler.fit_transform(dataset)
        
        train_data=data_training_array[0:data_training_len, :]
        x_train=[]
        y_train=[]
        
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])
            
        x_train,y_train=np.array(x_train),np.array(y_train)
        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        
        model1=load_model('Accurate.h5')
        
        test_data=data_training_array[data_training_len-60: ,:]
        x_test=[]
        y_test=dataset[data_training_len: ,:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i,0])
        
        x_test=np.array(x_test)
        x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        predictions=model1.predict(x_test)
        predictions=scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        
        train=data[:data_training_len]
        valid=data[data_training_len:]
        valid['Predictions'] = predictions
    
        fig2=plt.figure(figsize=(12,6))
        plt.plot(df.Close)
        plt.plot(valid[['Predictions']])
        plt.title('Actual Price vs Predicted Price')
        plt.xlabel('Date',fontsize=18)
        plt.xlabel('Closing Price',fontsize=18)
        plt.legend(['Actual Price','Predicted Price'])
        st.pyplot(fig2)
    
        last_60_days = data[-60:].values
        last_60_days_scaled = scaler.fit_transform(last_60_days)
        X_test1 = []
        X_test1.append(last_60_days_scaled)
        X_test1 = np.array(X_test1)
        X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
        pred_price = model1.predict(X_test1)
        pred_price = scaler.inverse_transform(pred_price)
    
        st.write(f'**Root Mean Squared Error is : {rmse}**')
        if close_value.empty:
            st.write(f"**{two_days_ago} Closing price was : Stock Market was closed!!!**")
        else :
            st.write(f'**{two_days_ago} Closing price was :  {close_value.iloc[0]}**')
        try:
            if df1.empty:
                st.write(f"**{yesterday} Closing price was : Stock Market was closed!!!**")
            else:
                todays_close_price = df1['Close'].iloc[0]
                st.write(f'**{yesterday} Closing price was :  {todays_close_price}**')
        except FileNotFoundError:
            st.write("**{yesterday} Closing price was : Stock Market was closed !!!**")
        st.write(f'**Todays Closing price might be : {pred_price[0][0]}**')
        st.dataframe(valid.tail(10))
    
