#src/utils/gemini_api.py
from google.generativeai import configure,  GenerativeModel
from dotenv import dotenv_values

config = dotenv_values(".env")
api_key = config["API_KEY"]

# configure(api_key=api_key)
model = GenerativeModel("gemini-2.0-flash")


