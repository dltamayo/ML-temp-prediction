#! usr/bin/env Python3

from KalshiClientsBaseV2 import ExchangeClient
import uuid
import pandas as pd
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
import os


def combine_weather_data(city_dict, path, date):
    recent_nws = pd.read_csv(f'{path}/weather_forecast/{city_dict['issue']}/{date}/recent_NWS.csv',
                                index_col=0)

    historical_om = pd.read_csv(f'{path}/weather_forecast/{city_dict['issue']}/{date}/historical_openmeteo.csv',
                                index_col=0)
    
    historical_ms = pd.read_csv(f'{path}/weather_forecast/{city_dict['issue']}/{date}/historical_meteostat.csv',
                                index_col=0)

    forecast_aw = pd.read_csv(f'{path}/weather_forecast/{city_dict['issue']}/{date}/forecast_accuweather.csv',
                                index_col=0)
    
    forecast_nws = pd.read_csv(f'{path}/weather_forecast/{city_dict['issue']}/{date}/forecast_NWS.csv',
                                index_col=0)

    forecast_vc = pd.read_csv(f'{path}/weather_forecast/{city_dict['issue']}/{date}/forecast_visualcrossing.csv',
                                index_col=0)
    
    all_pandas = [recent_nws, historical_om, historical_ms, forecast_aw, forecast_nws, forecast_vc]
    
    combined_df = pd.concat(all_pandas, ignore_index=True)

    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

    # Group by 'Date' and calculate the average temperature
    average_temperatures = combined_df.groupby('Date')['Temp'].mean().reset_index()
    average_temperatures['Temp'] = average_temperatures['Temp'].round(2)

    # Create separate columns for year, month, and date.
    average_temperatures['Year'] = pd.to_datetime(average_temperatures['Date']).dt.year
    average_temperatures['Month'] = pd.to_datetime(average_temperatures['Date']).dt.month
    average_temperatures['Day'] = pd.to_datetime(average_temperatures['Date']).dt.day

    return average_temperatures

def predict_weather(city_dict, dataset_date, predict_date):
    path = os.getcwd()
    data = combine_weather_data(city_dict, path, dataset_date)

    # Separate the date column into year, month, and day columns
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    data['Day'] = pd.to_datetime(data['Date']).dt.day

    # Extract features and split into training and test sets.
    X = data[['Year', 'Month', 'Day']]
    y = data['Temp'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Gradient Boosted Tree Regressor.
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    # Make predictions on the test set and compute MSE.
    gb_test_predictions = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_test_predictions)
    print("Gradient Boosted Test MSE:", round(gb_mse, 2))

    # Make prediction for date
    dt = datetime.strptime(predict_date, '%Y.%m.%d')    
    date_to_predict = [[dt.year, dt.month, dt.day]]

    # Ignore UserWarning temporarily.
    warnings.filterwarnings('ignore', category=UserWarning)
    gb_predicted_temperature = gb_model.predict(date_to_predict)
    # Reset warning filters.
    warnings.resetwarnings()

    print(f"Gradient Boosted Predicted temperature for {predict_date}:", round(gb_predicted_temperature[0], 2))
    return gb_predicted_temperature[0]

def select_ticker(ticker_column, number):
    bt_tuple = [(ticker[-2], float(ticker[-1])) for ticker in ticker_column.str.extract(r'([TB])(\d+(?:\.\d+)?)$').values]
    ticker_list = list(zip(ticker_column.tolist(), bt_tuple))

    # Separate 'T' and 'B' tickers.
    t_tickers = [t for t in ticker_list if t[1][0] == 'T']
    b_tickers = [t for t in ticker_list if t[1][0] == 'B']

    # If number is outside range of T tuples, return corresponding ticker.
    if 'T' in [t[1][0] for t in ticker_list]:
        max_t = max(t_tickers, key=lambda x: x[1][1])
        min_t = min(t_tickers, key=lambda x: x[1][1])
        if number > max_t[1][1] or number < min_t[1][1]:
            return max_t[0] if number > max_t[1][1] else min_t[0]

    # If number within T tuples, return ticker of corresponding B tuple.
    if 'B' in [t[1][0] for t in ticker_list]:
        for b_t in b_tickers:
            if abs(number - b_t[1][1]) <= 0.5 and number == int(number):
                return b_t[0]

    return None

def make_order(city, exchange_client, dataset_date, predict_date):
    # Get markets for high temperature of city.
    market_params = {'limit':50,
                        'series_ticker':city['series_ticker']}
    markets_response = exchange_client.get_markets(**market_params)
    
    # Filter active markets with provided date.
    formatted_date = datetime.strptime(predict_date, '%Y.%m.%d').strftime("%y%b%d").upper()
    climate_market = pd.DataFrame(data = markets_response['markets'])
    climate_market = climate_market[climate_market['ticker'].str.contains(formatted_date)]
    climate_market = climate_market[climate_market['status'] == 'active']

    # Make prediction for temperature and select corresponding ticker.
    predicted_temp = round(predict_weather(city, dataset_date, predict_date))
    ticker = select_ticker(climate_market['ticker'], predicted_temp)

    # Set order to expire at closing time of market.
    expire_time = climate_market.loc[climate_market['ticker'] == ticker, 'close_time'].iloc[0]
    expire_time = datetime.fromisoformat(expire_time.replace('Z', '+00:00'))
    expire_time = int(expire_time.timestamp())

    # Create order.
    order_params = {'ticker':ticker,
                    'client_order_id':str(uuid.uuid4()),
                    'type':'limit',
                    'action':'buy',
                    'side':'yes',
                    'count':1,
                    'yes_price':99, # yes_price = 100 - no_price
                    'no_price':None, # no_price = 100 - yes_price
                    'expiration_ts':expire_time,
                    'sell_position_floor':None,
                    'buy_max_cost':None}
    exchange_client.create_order(**order_params)

    print('Order successfully completed:')
    return order_params


if __name__ == "__main__":

    # Connect to Kalshi exchange client.
    with open('logins.json') as f:
        kalshi_login = json.load(f)    
    demo_email = kalshi_login['email']
    demo_password = kalshi_login['password']
    demo_api_base = "https://demo-api.kalshi.co/trade-api/v2"

    exchange_client = ExchangeClient(exchange_api_base = demo_api_base, email = demo_email, password = demo_password)

    # Check exchange status connection.
    print(exchange_client.get_exchange_status())

    # Make orders for each city.
    with open('city_config.json') as f:
        city_data = json.load(f)

    for city in city_data.keys():
        print(city)
        print(make_order(city_data[city], exchange_client, '2024.04.01', '2024.04.04'))
