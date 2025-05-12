import os
from typing import Optional, List, Dict, Any, Literal
import openai
from openai import OpenAI
import base64
from PIL import Image
import io
import json

class OpenAPI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI API connection
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No OpenAI API key provided")
        
        self.client = OpenAI(api_key=self.api_key)

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image using GPT-4 Vision
        Args:
            image_path (str): Path to the image file
            prompt (str): Question or prompt about the image
        Returns:
            str: Analysis of the image
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
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
                ],
                max_tokens=300
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("No content received from OpenAI")
            return content
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise

    def text_completion(self, prompt: str, model: str = "gpt-4", max_tokens: int = 1000) -> str:
        """
        Get text completion from OpenAI
        Args:
            prompt (str): Text prompt
            model (str): Model to use (default: gpt-4)
            max_tokens (int): Maximum number of tokens in the response
        Returns:
            str: Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("No content received from OpenAI")
            return content
        except Exception as e:
            print(f"Error getting text completion: {str(e)}")
            raise

    def get_recipe_suggestions(self, ingredients: List[str], dietary_restrictions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get recipe suggestions based on ingredients
        Args:
            ingredients (List[str]): List of available ingredients
            dietary_restrictions (List[str], optional): List of dietary restrictions
        Returns:
            Dict[str, Any]: Recipe suggestions and analysis
        """
        try:
            prompt = f"Given these ingredients: {', '.join(ingredients)}"
            if dietary_restrictions:
                prompt += f"\nAnd these dietary restrictions: {', '.join(dietary_restrictions)}"
            prompt += "\nSuggest 3 recipes that could be made with these ingredients. For each recipe, include:\n"
            prompt += "1. Name of the recipe\n2. List of ingredients needed\n3. Brief cooking instructions\n4. Estimated cooking time\n"
            prompt += "Format the response as a JSON object with 'recipes' as an array of recipe objects."

            response = self.text_completion(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                raise ValueError("Failed to parse recipe suggestions as JSON")
        except Exception as e:
            print(f"Error getting food suggestions: {str(e)}")
            raise
            
    def get_recipe(self, recipe_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a detailed recipe based on basic recipe information
        Args:
            recipe_info (Dict[str, Any]): Basic information about the recipe (name, ingredients, etc.)
        Returns:
            Dict[str, Any]: Detailed recipe information
        """
        try:
            # Create a premade prompt with the recipe information
            prompt = f"""
            Please provide a detailed recipe for {recipe_info.get('name', 'this dish')}.
            
            I have the following information about the recipe:
            - Name: {recipe_info.get('name', 'Not specified')}
            - Ingredients: {', '.join(recipe_info.get('ingredients', []))}
            - Cooking time: {recipe_info.get('cooking_time', 'Not specified')}
            
            Please provide:
            1. A brief description of the dish
            2. Detailed list of ingredients with quantities
            3. Step-by-step cooking instructions
            4. Nutritional information (approximate)
            5. Any tips or variations for this recipe
            
            Format the response as a JSON object with the following structure:
            {{
                "name": "Recipe name",
                "description": "Brief description",
                "ingredients": [
                    {{"name": "Ingredient name", "quantity": "amount", "unit": "measurement unit"}}
                ],
                "instructions": ["Step 1", "Step 2", ...],
                "nutritional_info": {{"calories": X, "protein": Y, "carbs": Z, "fat": W}},
                "tips": ["Tip 1", "Tip 2", ...],
                "variations": ["Variation 1", "Variation 2", ...]
            }}
            """
            
            response = self.text_completion(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                raise ValueError("Failed to parse recipe as JSON")
        except Exception as e:
            print(f"Error getting recipe: {str(e)}")
            raise