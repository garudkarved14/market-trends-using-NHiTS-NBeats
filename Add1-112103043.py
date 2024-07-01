import streamlit as st
import pandas as pd
import yfinance as yf  
from PIL import Image
import urllib.request
from bsedata.bse import BSE
import json
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.losses.pytorch import MAE
from neuralforecast import NeuralForecast

# Set the theme for seaborn
sns.set_theme()

# Add a title and an image
st.write("""
# Stock Market Analysis
""")

urllib.request.urlretrieve('https://cpb-us-w2.wpmucdn.com/u.osu.edu/dist/6/44792/files/2017/04/stock-market-3-21gyd1b.jpg', "stock_image")

image = Image.open("stock_image")

st.image(image, width=500)

# Create a sidebar header
st.sidebar.header('User Input')

# Create a function to get the users input
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2015-01-02")
    end_date = st.sidebar.text_input("End Date", "2020-01-02")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "GOOGL")
    return start_date, end_date, stock_symbol

# Create a function to get the proper company data and the proper timeframe
def get_data(symbol, start, end):
    # Load Data
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df = yf.download(symbol, start, end)
    return df

start, end, symbol = get_input()
df = get_data(symbol, start, end)

# Display the Sensex
today = date.today()
yesterday = today - timedelta(days = 10)
senstart = pd.to_datetime(yesterday)
senend = pd.to_datetime(today)

sen_df = yf.download("^BSESN", senstart, senend)

st.write("""
# Current Sensex Close Value
""")
cur_val = sen_df['Close'].iloc[-1]
st.write("# ", cur_val)

st.write("""
# BSE Sensex Close Price
""")
new_start = pd.to_datetime("2000-01-01")
new_sen_df = yf.download("^BSESN", new_start, senend)

st.line_chart(new_sen_df['Close'])

# BSE Top Runners and Top Losers
b = BSE()
tg = b.topGainers()
tg_data = json.dumps(tg)
tg_df = pd.read_json(tg_data)
tl = b.topLosers()
tl_data = json.dumps(tl)
tl_df = pd.read_json(tl_data)

st.write("""
# BSE Top Gainers
""")
st.write(tg_df)

st.write("""
# BSE Top Losers
""")
st.write(tl_df)
st.write("----------------------------------------------------------------------------------")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    color: red;
}
.norm-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Disclaimer : </p>', unsafe_allow_html=True)
st.markdown('<p class="norm-font">On Mobile, Open > at the top left for Company Specific Stock Details</p>', unsafe_allow_html=True)

st.write("# Stock Statistics of "+ symbol)

# Display the close price
st.header(symbol + " Close Price\n")
st.line_chart(df['Close'])

# Display the Volume
st.header(symbol + " Volume\n")
st.line_chart(df['Volume'])

# Display the statistics
st.header('Data Statistics')
st.write(df.describe())

@st.cache
def fetch_data(ticker):
    data = yf.download(ticker)
    close_prices = data[['Close']]
    close_prices = close_prices.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    close_prices['ds'] = pd.to_datetime(close_prices['ds'])
    close_prices = close_prices.set_index('ds')
    full_range = pd.date_range(start=close_prices.index.min(), end=close_prices.index.max(), freq='D')
    close_prices = close_prices.reindex(full_range).rename_axis('ds').reset_index()
    close_prices['y'] = close_prices['y'].interpolate()
    close_prices['unique_id'] = 'Stock'
    return close_prices

def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['ds'], data['y'], label='Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Over Time')
    plt.legend()
    plt.tight_layout()
    return plt

def run_prediction(data):
    train_end_date = data['ds'].iloc[-1] 
    Y_train_df = data[data.ds <= train_end_date]    
    horizon = 5

    model = NBEATS(
        h=horizon,
        input_size=20 * horizon,
        n_harmonics=2,
        n_polynomials=2,
        stack_types=['trend','seasonality', 'identity'],
        n_blocks=[1, 3, 1],
        mlp_units=[[512, 512], [512, 512],[512, 512]],
        activation='ReLU',
        shared_weights=False,
        loss=MAE(),
        max_steps=250,
        learning_rate=5e-4,
        num_lr_decays=2,
        batch_size=32,
        random_seed=1, 
    )
    nf = NeuralForecast(models=[model], freq='D')

    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(Y_train_df['ds'][-50:], Y_train_df['y'][-50:], label='Training Data')
    plt.plot(Y_hat_df['ds'], Y_hat_df['NBEATS'], label='NBEATS')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.tight_layout()
    return Y_hat_df, plt


st.title('Stock Price Forecasting App')

ticker = st.text_input('Enter Stock Ticker:', 'AAPL')

if st.button('Fetch Data'):
    data = fetch_data(ticker)
    st.pyplot(plot_data(data))

if st.button('Predict'):
    data = fetch_data(ticker)
    Y_hat_df, prediction_plot = run_prediction(data)  
    st.pyplot(prediction_plot)

    # Calculate tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow = tomorrow.strftime('%Y-%m-%d')

    # Get the prediction for tomorrow
    if tomorrow in Y_hat_df['ds'].dt.strftime('%Y-%m-%d').values:
        next_day_prediction = Y_hat_df[Y_hat_df['ds'].dt.strftime('%Y-%m-%d') == tomorrow]['NBEATS'].iloc[0]
        st.write("# ", "Tomorrow's Predicted Stock Price: ", next_day_prediction)
    else:
        st.write("No prediction available for tomorrow.")



# Add a tab for Gold Stock Analysis
st.title('Gold Stock Analysis')
gold_ticker = 'GC=F'

with st.expander("Gold Stock Data and Prediction"):
    if st.button('Load Gold Data', key='gold'):
        gold_data = fetch_data(gold_ticker)
        st.pyplot(plot_data(gold_data))

    if st.button('Predict Gold Price', key='gold_predict'):
        gold_data = fetch_data(gold_ticker)
        Y_hat_gold, gold_prediction_plot = run_prediction(gold_data)  
        st.pyplot(gold_prediction_plot)

        # Calculate tomorrow's date for gold prediction
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow = tomorrow.strftime('%Y-%m-%d')
        if tomorrow in Y_hat_gold['ds'].dt.strftime('%Y-%m-%d').values:
            next_day_gold_prediction = Y_hat_gold[Y_hat_gold['ds'].dt.strftime('%Y-%m-%d') == tomorrow]['NBEATS'].iloc[0]
            st.write("# ", "Tomorrow's Predicted Gold Price: ", next_day_gold_prediction)
        else:
            st.write("No prediction available for tomorrow for gold.")


#from forex_python.converter import CurrencyRates


# # Add a tab for Forex Exchange Rates
# st.title('Forex Exchange Rates')
# forex_pairs = [('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('AUD', 'USD')]
# currency_rates = CurrencyRates()

# with st.expander("Forex Rates"):
#     forex_data = {f"{pair[0]}/{pair[1]}": currency_rates.get_rate(pair[0], pair[1]) for pair in forex_pairs}
#     forex_df = pd.DataFrame(list(forex_data.items()), columns=['Currency Pair', 'Rate'])
#     st.write(forex_df)

# Add a tab for Forex Predictions using your model
st.title('Forex Predictions')

# List of popular forex tickers available in Yahoo Finance
forex_tickers = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X']

# Dropdown to select forex ticker
selected_forex_ticker = st.selectbox('Select Forex Ticker for Analysis:', forex_tickers)

# Expander for managing layout
with st.expander("Forex Data and Prediction"):
    if st.button('Load Forex Data', key='forex_load'):
        forex_data = fetch_data(selected_forex_ticker)
        st.pyplot(plot_data(forex_data))

    if st.button('Predict Forex Rates', key='forex_predict'):
        forex_data = fetch_data(selected_forex_ticker)
        Y_hat_forex, forex_prediction_plot = run_prediction(forex_data)
        st.pyplot(forex_prediction_plot)

        # Calculate tomorrow's date for forex prediction
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow = tomorrow.strftime('%Y-%m-%d')
        if tomorrow in Y_hat_forex['ds'].dt.strftime('%Y-%m-%d').values:
            next_day_forex_prediction = Y_hat_forex[Y_hat_forex['ds'].dt.strftime('%Y-%m-%d') == tomorrow]['NBEATS'].iloc[0]
            st.write("# ", "Tomorrow's Predicted Forex Rate: ", next_day_forex_prediction)
        else:
            st.write("No prediction available for tomorrow for this forex rate.")
