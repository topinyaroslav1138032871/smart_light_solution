import requests
from config import *


def getDataWeather(city: str) -> dict:
    url = f"https://api.weatherapi.com/v1/current.json?key={APIKEY}&q={city}&aqi=no&lang=ru"

    response = requests.request("GET", url)

    isDay = True
    cloud = response.json()['current']['cloud']

    if (response.json()['current']['is_day'] == 0):
        isDay = False

    return { "isDay": isDay, "cloudProc": cloud}

print(getDataWeather('Курск'))