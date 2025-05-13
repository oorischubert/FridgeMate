import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from typing import Optional, List
from fridge import Fridge
from food import Food, FOOD_TYPES

FIREBASE_APP_NAME = "fridgemate123"
FIREBASE_CREDENTIALS_PATH = "/Users/oorischubert/certificates/fridgemate123_cert.json"

class FirebaseUploader:
    _app_initialized = False 

    def __init__(self, credentials_path=FIREBASE_CREDENTIALS_PATH):
        """Initialize Firebase connection using credentials."""
        self.db = None
        
        # Initialize Firebase Admin SDK only once
        if not FirebaseUploader._app_initialized:
            try:
                if not os.path.exists(credentials_path):
                        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
                cred = credentials.Certificate(credentials_path)
                print(f"Using credentials file: {credentials_path}")
               
                firebase_admin.initialize_app(cred, name=FIREBASE_APP_NAME) 
                FirebaseUploader._app_initialized = True
                print(f"Firebase app '{FIREBASE_APP_NAME}' initialized.")

            except (ValueError, FileNotFoundError) as e:
                 print(f"Firebase Init Error: {e}")
                 raise
            except Exception as e:
                print(f"Unexpected error initializing Firebase: {e}")
                raise

        # Get Firestore client for the initialized app
        try:
             app_instance = firebase_admin.get_app(name=FIREBASE_APP_NAME)
             self.db = firestore.client(app=app_instance)
        except Exception as e:
             print(f"Error getting Firestore client: {e}")
             raise # Cannot proceed without Firestore client
         
    def get_fridge(self, fridge: Fridge) -> Fridge:
        if not self.db:
            print("Error: Firestore client not initialized.")
            return fridge

        try:
            fridge_ref = self.db.collection('main').document('fridge')
            fridge_data = fridge_ref.get().to_dict()

            if fridge_data:
                fridge.from_dict(fridge_data)
                print("Fridge data retrieved successfully.")
            else:
                print("No fridge data found in Firestore.")

        except Exception as e:
            print(f"Error retrieving fridge data: {e}")

        return fridge

    def set_fridge(self, fridge: Fridge) -> bool:
        if not self.db:
            print("Error: Firestore client not initialized.")
            return False

        try:
            fridge_ref = self.db.collection('main').document('fridge')
            fridge_ref.set(fridge.to_dict())
            print("Fridge data uploaded successfully.")
            return True

        except Exception as e:
            print(f"Error uploading fridge data: {e}")
            return False

    def upload_food(self, food: Food) -> Optional[str]:
        """Upload food data (without image) to Firebase Firestore."""
        if self.db is None:
             print("Error: Firestore client not initialized.")
             return None
             
        try:
            food_data = food.to_dict()

            # Ensure expirationDate is Firestore-compatible
            if isinstance(food_data['expirationDate'], datetime):
                 food_data['expirationDate'] = food_data['expirationDate'] 

            # Upload to Firestore collection: main/fridge/foods
            doc_ref = self.db.collection('main').document("fridge").collection("foods").document(food.name)
            doc_ref.set(food_data)
            
            print(f"Food '{food.name}' uploaded to Firestore. ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            print(f"Error uploading food '{food.name}' to Firestore: {e}")
            return None 

    # --- get_foods, update_food_status methods (modified slightly for clarity) ---

    def get_foods(self, in_fridge: Optional[bool] = None, limit: int = 100) -> List[Food]:
        """Get food items from Firebase."""
        if not self.db:
             print("Error: Firestore client not initialized.")
             return []
        try:
            # Path: /main/fridge/foods
            query = self.db.collection('main/fridge/foods') 
            
            if in_fridge is not None:
                query = query.where('inFridge', '==', in_fridge)
            
            docs = query.limit(limit).stream()
            
            foods_list = []
            for doc in docs:
                 try:
                      foods_list.append(Food.from_dict(doc.to_dict()))
                 except Exception as parse_e:
                      print(f"Error parsing document {doc.id}: {parse_e}")
            return foods_list
        except Exception as e:
            print(f"Error retrieving foods: {e}")
            raise

    def update_food_status(self, food_id: str, in_fridge: bool) -> bool:
        """Update the fridge status of a food item."""
        if not self.db:
             print("Error: Firestore client not initialized.")
             return False
        try:
            doc_ref = self.db.collection('main/fridge/foods').document(food_id)
            doc_ref.update({
                'inFridge': in_fridge,
            })
            print(f"Food status updated for ID: {food_id}")
            return True
        except Exception as e:
            print(f"Error updating food status for {food_id}: {e}")
            return False

    def close(self):
        """Close Firebase connection."""
        try:
            if FirebaseUploader._app_initialized:
                 app_instance = firebase_admin.get_app(name=FIREBASE_APP_NAME)
                 firebase_admin.delete_app(app_instance)
                 FirebaseUploader._app_initialized = False 
                 print(f"Firebase app '{FIREBASE_APP_NAME}' closed.")
        except Exception as e:
            print(f"Error closing Firebase connection: {e}")

# --- Main block for testing ---
if __name__ == "__main__":
    print("--- Testing FirebaseUploader (Food Data Upload Only) ---")

    # --- Configuration ---
    # Provide path to your Firebase Service Account Key JSON file
    test_credentials_path = FIREBASE_CREDENTIALS_PATH
    
    # --- Test Execution ---
    uploader = None 
    try:
        print("\nInitializing FirebaseUploader...")
        # Initialize uploader with credentials path only
        uploader = FirebaseUploader(credentials_path=test_credentials_path)
        print("Initialization successful.")

        print("\nCreating sample food item...")
        sample_food = Food(
            name="avocado",
            expirationDate=datetime.now() + timedelta(days=7),
            count=2,
            inFridge=True,
            description="stupid avocado",
            foodType=FOOD_TYPES[0]
        )
        
        sample_fridge = Fridge(
            uid="123",
            humidity=1,
            temperature=34,
            isOpen=False
        )
        
        print(f"Sample food created: {sample_food.name}")

        print(f"\nAttempting to upload '{sample_food.name}' data...")
        document_id = uploader.upload_food(sample_food) 
        #document_id = uploader.set_fridge(sample_fridge)

        if document_id:
            print(f"\n--- Upload Test SUCCESS ---")
            print(f"Food item data uploaded with Document ID: {document_id}")
            
            # Optional: Test fetching data
            print("\nFetching last 5 food items...")
            fetched_foods = uploader.get_foods(limit=5)
            if fetched_foods:
                 print(f"Fetched {len(fetched_foods)} items:")
                 for idx, food in enumerate(fetched_foods):
                      print(f"  {idx+1}. Name: {food.name}, Expires: {food.expirationDate}, InFridge: {food.inFridge}, foodType: {food.foodType}")
            else:
                 print("No food items fetched or an error occurred.")
        else:
            print("\n--- Upload Test FAILED ---")

    except (ValueError, FileNotFoundError) as e:
        print(f"\nConfiguration/File Error: {e}")
        print(f"Please ensure '{test_credentials_path}' is the correct path to your service account key.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {type(e).__name__} - {e}")
    finally:
        # if uploader: uploader.close() # Optional cleanup
        print("\n--- Firebase Test Finished ---")