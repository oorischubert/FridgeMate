import os
from typing import Optional, List, Dict, Any, Literal
import openai
from openai import OpenAI
import base64
from PIL import Image
import io
import json
from datetime import datetime, timedelta
from food import Food, FOOD_TYPES
from firebase import FirebaseUploader

LABEL_PROMPT = 'What is the person holding in their hand? Respond in json format with label and whether it is a food item or not in format {"label": "item_label", "food": "bool", "count": "int_numberofitems", "description": "short_descriptionof_item", type: "food_type:} it not holding anything, both the description and label should be "none", count 0 and is_food should be false. type must exist in %s or be none if not food.'%FOOD_TYPES
MODEL = 'gpt-4.1-mini'

# https://platform.openai.com/playground/prompts?models=gpt-4.1-mini

class OpenAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No OpenAI API key provided")
        
        self.client = OpenAI(api_key=self.api_key)

    def getLabel(self, image_path: str, prompt: str = LABEL_PROMPT) -> Dict[str, Any]:
        try:
            # Read and encode the image
            with Image.open(image_path) as img:
                # Convert image to RGB if it's not (e.g., if it's RGBA or P)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save image to a bytes buffer
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG") # Or PNG, depending on your preference
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            response = self.client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"}, # Request JSON output
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # The response content should be a JSON string, so parse it
            if response.choices and response.choices[0].message.content:
                json_response = json.loads(response.choices[0].message.content)
                return json_response
            else:
                raise ValueError("Failed to get a valid response from OpenAI API")

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return {"error": f"Image file not found at {image_path}"}
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return {"error": f"OpenAI API Error: {str(e)}"}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Raw response: {response.choices[0].message.content if response.choices and response.choices[0].message else 'No content'}")
            return {"error": "Failed to decode JSON response from API"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": f"An unexpected error occurred: {str(e)}"}
        
if __name__ == "__main__":
    print("Testing OpenAPI getLabel function...")

    test_image_path = "test_images/test_image1.jpg" 
    
    server = FirebaseUploader()

    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}. Please provide a valid image for testing.")
        exit(1)

    if os.path.exists(test_image_path):
        try:
            api = OpenAPI() 

            print(f"\n--- Test 1: Using default prompt for image: {test_image_path} ---")
            result1 = api.getLabel(image_path=test_image_path)
            print("Response from getLabel (default prompt):")
            print(json.dumps(result1, indent=2))
            print("\n--- Parsed Information ---")
             
            if result1 and isinstance(result1, dict) and 'error' not in result1:
                # Use .get() for safer access in case keys are missing
                item_label = result1.get('label', 'Label not found')
                is_food = result1.get('food', 'false')
                count = result1.get('count','0')
                item_description = result1.get('description', 'Description not found')
                foodType = result1.get('type', 'error_parsing')
                                
                try:
                    is_food = bool(is_food)
                except Exception as e:
                    print(f"Error converting {is_food} to bool!")
                try:
                    count = int(count)
                except Exception as e:
                    print(f"Error converting {count} to int!")
                    
                food = Food(name=item_label, description=item_description, count=count, expirationDate=datetime.now()+timedelta(days=7), inFridge=True, foodType=foodType)
                
                print(food)
                
                #server.upload_food(food=food)
                
                # Check if the food type exists in FOOD_TYPES
                food_type_exists = foodType in FOOD_TYPES
                print(f"Type: {foodType}, exists in typeList = {food_type_exists}")                 
            

        except ValueError as ve:
            print(f"Error during testing: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during testing: {e}")
    else:
        print(f"Test image '{test_image_path}' not found. Skipping tests.")

    print("\nTesting finished.")