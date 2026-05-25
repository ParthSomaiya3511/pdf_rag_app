import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_gemini(prompt):

    model = genai.GenerativeModel(
        model_name="models/gemini-pro"
    )

    response = model.generate_content(prompt)

    return response.text