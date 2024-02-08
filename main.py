import yfinance as yf
import streamlit as st
from datetime import date
from prophet import Prophet
import pandas as pd

from plotly import graph_objs as go
from prophet.plot import plot_plotly

from tickers import get_ticker

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Stock Price Prediction App')

stocks = ([ticker for ticker in get_ticker()])
stock = st.selectbox('Select any stock', stocks)

n_years = st.slider('Years of prediction', 1, 4)
period = n_years*365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Load data...')
# data = load_data(f'{stock}.NS')
data = load_data('AAPL')
data_load_state.text('Loading data... Done!')


st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.subheader('forecast data')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.subheader('Accuracy Measure')
# fig2 = model.plot_components(forecast)
# st.write(fig2)

from prophet.diagnostics import cross_validation
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
# st.write(df_cv.head())

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
st.write(df_p[['horizon', 'mape']].head())

from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
st.plotly_chart(fig)





