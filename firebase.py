import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
import json
from typing import Optional, List
from food import Food

class FirebaseUploader:
    def __init__(self, credentials_path=None, storage_bucket=None):
        """
        Initialize Firebase connection
        Args:
            credentials_path (str): Path to Firebase credentials JSON file
            storage_bucket (str): Firebase Storage bucket name
        """
        try:
            if credentials_path:
                cred = credentials.Certificate(credentials_path)
            else:
                # Try to get credentials from environment variable
                cred_json = os.getenv('FIREBASE_CREDENTIALS')
                if not cred_json:
                    raise ValueError("No Firebase credentials provided")
                cred = credentials.Certificate(json.loads(cred_json))
            
            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })
            self.db = firestore.client()
            self.bucket = storage.bucket() if storage_bucket else None
            print("Firebase connection initialized successfully")
        except Exception as e:
            print(f"Error initializing Firebase: {str(e)}")
            raise

    def upload_food(self, food: Food, image_path: Optional[str] = None) -> str:
        """
        Upload food data to Firebase
        Args:
            food (Food): Food object to upload
            image_path (str, optional): Path to the food image file
        Returns:
            str: Document ID of the uploaded food
        """
        try:
            # Upload image if provided
            image_url = None
            if image_path and self.bucket:
                image_url = self._upload_image(image_path, food.name)

            # Prepare food data
            food_data = food.to_dict()
            food_data.update({
                'timestamp': datetime.now(),
                'image_url': image_url
            })

            # Upload to Firestore
            doc_ref = self.db.collection('main').document("fridge").collection("foods").document()
            doc_ref.set(food_data)
            
            print(f"Food uploaded successfully with ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            print(f"Error uploading food: {str(e)}")
            raise

    def _upload_image(self, image_path: str, food_name: str) -> str:
        """
        Upload image to Firebase Storage
        Args:
            image_path (str): Path to the image file
            food_name (str): Name of the food item
        Returns:
            str: Public URL of the uploaded image
        """
        try:
            if not self.bucket:
                raise ValueError("Storage bucket not initialized")

            # Create a unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"foods/{food_name}_{timestamp}.jpg"
            
            # Upload the file
            blob = self.bucket.blob(filename)
            blob.upload_from_filename(image_path)
            
            # Make the file publicly accessible
            blob.make_public()
            
            return blob.public_url
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            raise

    def get_foods(self, in_fridge: Optional[bool] = None, limit: int = 10) -> List[Food]:
        """
        Get food items from Firebase
        Args:
            in_fridge (bool, optional): Filter by fridge status
            limit (int): Number of items to retrieve
        Returns:
            List[Food]: List of Food objects
        """
        try:
            query = self.db.collection('main').document("fridge").collection("foods")
            
            if in_fridge is not None:
                query = query.where('inFridge', '==', in_fridge)
            
            foods = query.order_by('timestamp', direction='DESCENDING')\
                .limit(limit)\
                .stream()
            
            return [Food.from_dict(doc.to_dict()) for doc in foods]
        except Exception as e:
            print(f"Error retrieving foods: {str(e)}")
            raise

    def update_food_status(self, food_id: str, in_fridge: bool) -> None:
        """
        Update the fridge status of a food item
        Args:
            food_id (str): ID of the food document
            in_fridge (bool): New fridge status
        """
        try:
            self.db.collection('main').document("fridge").collection("foods").document(food_id).update({
                'inFridge': in_fridge,
                'timestamp': datetime.now()
            })
            print(f"Food status updated successfully for ID: {food_id}")
        except Exception as e:
            print(f"Error updating food status: {str(e)}")
            raise

    def close(self):
        """
        Close Firebase connection
        """
        try:
            firebase_admin.delete_app(firebase_admin.get_app())
            print("Firebase connection closed successfully")
        except Exception as e:
            print(f"Error closing Firebase connection: {str(e)}")
            raise
