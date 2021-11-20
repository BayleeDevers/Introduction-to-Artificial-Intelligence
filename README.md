
# Usage
- The first time this is set up you will need to create a virtual environment (If you use anaconda this can be done using `conda create --name intro2ai python=3.7`).

- Activate the environment using `conda activate intro2ai`

- Run `pip install -r requirements.txt` to make sure the dependencies are installed.

- To use the YouTube API & Bigquery API, you will need to set the environment variable %I2AI_API_KEY% with the secret key that I can provide. 
You can set this by running: `setx I2AI_API_KEY <ask me for what goes here>`
Alternatively a secrets/ subdirectory with the neccessary .json key files. The reddit data required is also from an API although no key is necessary to run this.

- Once the two csvs (blocks.csv & reddi_posts.csv) have been produced using the 2 APIs a third csv is needed. This contains the main market data about the BTC/USD pair and can be downloaded from.
https://www.kaggle.com/mczielinski/bitcoin-historical-data/download

- `python convert.py -f <file path to bitstampUSD...csv>` will prepare the cache.csv which contains the data aggregated into time intervals

- Finally any of `python lstm_model.py` or `python lstm_clf_model.py` or `python dense_model.py` can be run to train and evaluate the model's performance

