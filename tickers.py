import pandas as pd

def get_ticker():
    df = pd.read_csv('ind_nifty50list.csv')
    tickers = df['Symbol']
    return tickers
