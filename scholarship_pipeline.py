pip install openai pandas python-dotenv

import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load the scholarship CSV
try:
    scholarship_df = pd.read_csv("scholarships.csv")
    print("Scholarship CSV loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading the scholarship CSV: {e}")

def refine_prompt(user_input):
    """
    Refine the user's input into a structured and detailed prompt using OpenAI's GPT model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at refining prompts specifically for finding scholarships and grants."},
                {"role": "user", "content": f"Please refine this query to be more exhaustive for finding scholarships: {user_input}"}
            ],
        )
        refined_prompt = response.choices[0].message.content.strip()
        return refined_prompt
    except Exception as e:
        print(f"Error refining prompt: {e}")
        return None

def search_scholarships(refined_prompt, df):
    """
    Search the scholarship DataFrame using the refined prompt.
    """
    try:
        # Perform a case-insensitive keyword search in the 'Content' column
        matching_scholarships = df[df['Content'].str.contains(refined_prompt, case=False, na=False)]

        # Return the top 5 matches or a message if no matches are found
        if not matching_scholarships.empty:
            return matching_scholarships.head(5)
        else:
            return pd.DataFrame({"Message": ["No matching scholarships found."]})
    except Exception as e:
        print(f"Error searching scholarships: {e}")
        return pd.DataFrame({"Message": [f"Error: {e}"]})

def run_pipeline(user_query):
    """
    End-to-end pipeline: Refine prompt, search scholarships, and return results.
    """
    refined_prompt = refine_prompt(user_query)
    if refined_prompt:
        print(f"Refined Prompt: {refined_prompt}")  # Debugging log
        search_results = search_scholarships(refined_prompt, scholarship_df)
        return search_results
    else:
        return pd.DataFrame({"Message": ["Failed to refine the prompt."]})
