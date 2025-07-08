# weather_tool.py
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# LLM model
llm = Ollama(model="llama3")

# Static place data
INDOOR_PLACES = {
    "dubai": [
        "Dubai Mall", "IMG Worlds of Adventure", "Dubai Aquarium", "Dubai Opera", "Museum of the Future"
    ],
    "abu dhabi": [
        "Yas Mall", "Louvre Abu Dhabi", "Marina Mall", "Qasr Al Hosn", "Warner Bros World"
    ],
    "sharjah": [
        "Sharjah Aquarium", "Mega Mall", "Sharjah Science Museum", "Al Qasba", "Sharjah Art Foundation"
    ]
}

OUTDOOR_PLACES = {
    "dubai": [
        "Burj Khalifa", "Jumeirah Beach", "Dubai Marina Walk", "Miracle Garden", "Desert Safari"
    ],
    "abu dhabi": [
        "Corniche Beach", "Sheikh Zayed Grand Mosque", "Ferrari World (outdoor parts)", "Al Ain Zoo"
    ],
    "sharjah": [
        "Al Noor Island", "Sharjah Beach", "Al Majaz Waterfront", "Safari Park", "Flag Island"
    ]
}

# LLM prompt
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Based on the weather info, decide whether indoor or outdoor activities are better for {city}. Then suggest a few suitable places from the lists below.

Indoor places: {indoor}
Outdoor places: {outdoor}

Weather:
- Temperature: {temp}°C
- Wind Speed: {wind} km/h

Answer:
""")

rag_chain = LLMChain(llm=llm, prompt=prompt_template)

def get_weather(city: str) -> str:
    city = city.strip().lower()

    city_coords = {
        "dubai": (25.276987, 55.296249),
        "abu dhabi": (24.4539, 54.3773),
        "sharjah": (25.3463, 55.4209),
    }

    latitude, longitude = city_coords.get(city, city_coords["dubai"])

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&hourly=temperature_2m&current_weather=true"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        weather = data.get("current_weather")

        if not weather:
            return f"Weather data not available for {city.title()}."

        temp = weather["temperature"]
        wind = weather["windspeed"]

        indoor_list = ", ".join(INDOOR_PLACES.get(city, INDOOR_PLACES["dubai"]))
        outdoor_list = ", ".join(OUTDOOR_PLACES.get(city, OUTDOOR_PLACES["dubai"]))

        inputs = {
            "temp": temp,
            "wind": wind,
            "city": city.title(),
            "indoor": indoor_list,
            "outdoor": outdoor_list
        }

        return rag_chain.run(inputs)

    except Exception as e:
        return f"Error fetching weather: {e}"
