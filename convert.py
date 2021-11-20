import numpy as np
import pandas as pd
from datetime import datetime as dt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='Path to csv file for input in conversion')
args = parser.parse_args()

def processna(data):
    # Fix the data for candles where there are no trades, NaN's replaced with 0's and Open and Close prices are the same
    data['Volume_(BTC)'].fillna(value=0, inplace=True)
    data['Volume_(Currency)'].fillna(value=0, inplace=True)
    data['Weighted_Price'].fillna(value=0, inplace=True)

    # next we need to fix the OHLC (open high low tta which is a continuous timeseries so
    # lets fill forwards those values...
    data['Open'].fillna(method='ffill', inplace=True)
    data['High'].fillna(method='ffill', inplace=True)
    data['Low'].fillna(method='ffill', inplace=True)
    data['Close'].fillna(method='ffill', inplace=True)

def group_by_timeframe(data, start, end, freq='3600s'):
    '''
    Takes a subset of data corresponding to that which is between start & end
    and aggregates subset into bins with interval length equal to freq
    '''
    aggregate = data[(data['timestamp'] >= start) & (data['timestamp'] <= end)] \
                .groupby([pd.Grouper(key='timestamp', freq=freq)])
    return aggregate

price = pd.read_csv(args.file, index_col=0)
price.rename(columns={'timestamp': 'timestamp'})
blocks = pd.read_csv('data/blocks.csv')
posts = pd.read_csv('data/reddit_posts.csv', header=None)

posts = posts.rename(columns={0: 'timestamp', 1: 'id', 2: 'author', 3: 'domain', 4: 'num_comments', 5: 'title'})
posts['title'] = posts['title'].astype(str)

new = pd.DataFrame({'Open': [],
                    'High': [],
                    'Low': [],
                    'Close': [],
                    'Volume_(BTC)': [],
                    'Volume_(Currency)': [],
                    'Weighted_Price': []})

for ts in range(price.index[0], price.index[-1] + 1, 60):
    #print(ts, ts in price.index)
    if ts not in price.index:
        new.loc[ts] = pd.Series({'Open': np.nan,
                                'High': np.nan,
                                'Low': np.nan,
                                'Close': np.nan,
                                'Volume_(BTC)': np.nan,
                                'Volume_(Currency)': np.nan,
                                'Weighted_Price': np.nan})
price = pd.concat([price, new])

# Construct comparable timestamps

posts['timestamp'] = posts['timestamp'].map(lambda x: dt.fromtimestamp(float(x)))
blocks['timestamp'] = blocks['timestamp'].map(lambda x: dt.fromisoformat(x).replace(tzinfo=None))
price['timestamp'] = price.index.map(lambda x: dt.fromtimestamp(float(x)))

# Start and end data at an appropriate timescale (take subset of total available data)
start = dt(2016, 1, 12, 0, 0, 0, 0)
end = dt(2020, 11, 11, 0, 0, 0, 0)

processna(price)

# Aggregate into less frequent intervals
price_hourly = group_by_timeframe(price, start, end).first().reset_index()
blocks_hourly = group_by_timeframe(blocks, start, end).first().reset_index()
posts_hourly = group_by_timeframe(posts, start, end)

titles_hourly = posts_hourly['title'].agg(lambda x: ' '.join(x)).reset_index()
num_comments_hourly = posts_hourly['num_comments'].sum().reset_index()

processna(price_hourly)

price_hourly = price_hourly.set_index('timestamp')
blocks_hourly = blocks_hourly.set_index('timestamp')
titles_hourly = titles_hourly.set_index('timestamp')
num_comments_hourly = num_comments_hourly.set_index('timestamp')

out_df = price_hourly.join(blocks_hourly).join(titles_hourly).join(num_comments_hourly)

out_df.to_csv('data/cache.csv')