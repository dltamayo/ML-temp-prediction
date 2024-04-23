#! usr/bin/env Python3

import re
import os
import sys
import json

import pandas as pd
from datetime import datetime, timedelta

import requests
import requests_cache
from retry_requests import retry
from bs4 import BeautifulSoup
from meteostat import Point, Daily
import openmeteo_requests

def meteostat(city, path):
    # Set start and end points for data collection.
    today = datetime.today()
    one_day_ago = today - timedelta(days=1)
    five_years_ago = datetime((today.year - 5), today.month, today.day)

    data = Point(city['latitude'], city['longitude'])
    data = Daily(data, five_years_ago, one_day_ago)
    data = data.fetch()

    # Convert Celsius to Fahrenheit; set names for Date and Max Temp.
    data.reset_index(inplace=True)
    celsius_to_fahrenheit = round((data['tmax'] * 9/5) + 32, 2)
    data.insert(1, 'Temp', celsius_to_fahrenheit)
    data.rename(columns={'time': 'Date'}, inplace=True)

    # Save data to csv.
    filename = f'{path}/historical_meteostat.csv'
    data.to_csv(filename)

    return f'Meteostat data saved to {filename} successfully'

def open_meteo(city, path):
    # Set up Open-Meteo API client with cache and retry on error.
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # Set start and end points for data collection.
    today = datetime.today()
    one_day_ago = today - timedelta(days=1)
    one_day_ago = one_day_ago.strftime('%Y-%m-%d')
    five_years_ago = datetime((today.year - 5), today.month, today.day)
    five_years_ago = five_years_ago.strftime('%Y-%m-%d')

    # List required weather variables for API call.
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': city['latitude'],
        'longitude': city['longitude'],
        'start_date': str(five_years_ago),
        'end_date': str(one_day_ago),
        'daily': 'temperature_2m_max',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'America/New_York'
    }

    # Retrieve and process daily data.
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()

    daily_data = {
        'date': pd.date_range(
            start=pd.to_datetime(daily.Time(), unit='s', utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit='s', utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive='left'
        )
    }
    daily_data['temperature_2m_max'] = daily_temperature_2m_max
    daily_dataframe = pd.DataFrame(data=daily_data)
    
    # Set names for Date and Max Temp.
    daily_dataframe.insert(0, 'Date', daily_dataframe['date'].dt.strftime('%Y-%m-%d'))
    daily_dataframe['Temp'] = daily_dataframe['temperature_2m_max'].round(2)
    daily_dataframe.drop(columns=['date', 'temperature_2m_max'], inplace=True)

    # Save data to csv.
    filename = f'{path}/historical_openmeteo.csv'
    daily_dataframe.to_csv(filename, index=False)

    return f'Open-Meteo data saved to {filename} successfully.'

def recent_forecast(city, num_pages, path):
    site,issue = city['site'],city['issue']
    url = f'https://forecast.weather.gov/product.php?site={site}&issuedby={issue}&product=CLI&format=TXT'

    data = []
    # Loop over previous pages of climatological report.
    for i in range(num_pages):
        url_ver = url + f'&version={i+1}&glossary=0'

        # Send a GET request to the URL.
        response = requests.get(url_ver)

        if response.status_code == 200:
            # Parse the HTML content of the page.
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find glossaryProduct element containing weather data.
            glossary_product_element = soup.find('pre', {'class': 'glossaryProduct'})
            
            if glossary_product_element:
                # Extract the text from the element.
                glossary_product_text = glossary_product_element.get_text()
                
                # Extract maximum temperature from climatological report.
                max_temp_match = re.search(r'TEMPERATURE\s\(F\)\s+([A-Z]+)\s+MAXIMUM\s+(\d+)', glossary_product_text)
                if not max_temp_match:
                    continue

                # Do not extract report of previous day, to avoid duplicate values.
                elif (max_temp_match.group(1) == 'YESTERDAY'):
                    continue

                # Retrieve date of report.
                date_regex = re.search(r'(\b(?:\w+)\s+\d{1,2}\s+\d{4}\b)', glossary_product_text)
                data.append({'Date':date_regex.group(1), 'Temp':max_temp_match.group(2)})

            else:
                print('No glossaryProduct text found on the page.')
                return
        else:
            print('Failed to retrieve the web page. Status code:', response.status_code)
            return
        
    data = pd.DataFrame(item for item in data)

    # Keep the most recent date for each climatological report.
    data = data.drop_duplicates(subset='Date', keep='first', ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'], format='%b %d %Y')

    # Reformat 'Date' column to YYYY-MM-DD format.
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data.iloc[::-1].reset_index(drop=True)
    data['Temp'] = data['Temp'].apply(float)

    # Save data to csv.
    filename = f'{path}/recent_NWS.csv'
    data.to_csv(filename)

    return f'Recent NWS data saved to {filename} successfully.'

def accuweather(city, path):
    with open('logins.json') as f:
        config = json.load(f)
        api_key = config['accuweather_key']

    api_key = 'bGEVjV79RL2adwunGMvtasFTS4UHpxpX'

    # Define the API endpoint and parameters
    url = f'http://dataservice.accuweather.com/forecasts/v1/daily/5day/{city['location_key']}'
    params = {
        'apikey': api_key,
        'details': False
    }

    # Make the GET request.
    forecast_response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if forecast_response.status_code == 200:
        forecast = forecast_response.json()
    else:
        print('Error:', forecast_response.status_code)
        return

    # Set names for Date and Max Temp.
    data = pd.DataFrame.from_dict(pd.json_normalize(forecast['DailyForecasts']), orient='columns')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data.drop(columns=['Sources', 'MobileLink', 'Link'], inplace=True)
    data.rename(columns={'Temperature.Maximum.Value': 'Temp'}, inplace=True)

    # Save data to csv.
    filename = f'{path}/forecast_accuweather.csv'
    data.to_csv(filename)

    return f'AccuWeather data saved to {filename} successfully.'

def visualcrossing(city, path):
    # Set API key for data call.
    with open('logins.json') as f:
        config = json.load(f)
        weather_crossing_key = config['visualcrossing_key']

    response = requests.request('GET', f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city['latitude']}%2C{city['longitude']}?unitGroup=us&include=days%2Cfcst&key={weather_crossing_key}&contentType=json')
    if response.status_code != 200:
        print('Unexpected Status code:', response.status_code)
        sys.exit() 
    jsonData = response.json()

    # Set names for Date and Max Temp.
    data = pd.DataFrame.from_dict(pd.json_normalize(jsonData['days']), orient='columns')
    data.rename(columns={'tempmax': 'Temp',
                         'datetime': 'Date'}, inplace=True)

    # Save data to csv.
    filename = f'{path}/forecast_visualcrossing.csv'
    data.to_csv(filename)

    return f'Visual Crossing data saved to {filename} successfully.'

def weather_gov(city, path):
    # Retrieve nearest Weather Forecast Office (WFO) based on latitude and longitude.
    response0 = requests.request('GET', f'https://api.weather.gov/points/{city['latitude']},{city['longitude']}')
    if response0.status_code != 200:
        print('Unexpected Status code:', response0.status_code)
        sys.exit()

    # Retrieve forecast data from WFO.
    response1 = requests.request('GET', response0.json()['properties']['forecast'])
    if response1.status_code != 200:
        print('Unexpected Status code:', response1.status_code)
        sys.exit()
    data = pd.DataFrame.from_dict(pd.json_normalize(response1.json()['properties']['periods']), orient='columns')

    # Format date and maximum temperature columns.
    data.insert(0, 'Date', pd.to_datetime(data['startTime']))
    data = data.loc[data['Date'].dt.date != pd.to_datetime('today').date()]
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data.drop(columns='number', inplace=True)
    
    # Set names for Date and Max Temp.
    data.rename(columns={'temperature': 'Temp'}, inplace=True)

    # For each day, select the maximum temperature.
    max_temps = data.groupby('Date')['Temp'].max()
    filtered_data = data[data.apply(lambda x: x['Temp'] == max_temps.loc[x['Date']], axis=1)]
    filtered_data.reset_index(drop=True, inplace=True)

    # Save data to csv.
    filename = f'{path}/forecast_NWS.csv'
    data.to_csv(filename)

    return f'National Weather Service data saved to {filename} successfully.'

def collect_weather_data(city, path):
    print('===recent===')
    print(recent_forecast(city, 50, path))
    print()

    print('===historical===')
    print(open_meteo(city, path))
    print(meteostat(city, path))
    print()

    print('===forecast===')
    print(accuweather(city, path))
    print(weather_gov(city, path))
    print(visualcrossing(city, path))

    return 'Data collected.'


if __name__ == "__main__":
    today = datetime.today()
    directory = os.getcwd()

    with open('city_config.json') as f:
        city_data = json.load(f)

    for key in city_data.keys():
        print(key)
        city = city_data[key]

        # Create directory for saving data csv.
        path = f'{directory}/weather_forecast/{city['issue']}/{today.strftime('%Y.%m.%d')}'
        os.makedirs(path, exist_ok=True)
        
        print(collect_weather_data(city, path))
        print()
