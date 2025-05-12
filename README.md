# FridgeMate 🍎

FridgeMate is a smart food management system that helps you track your food inventory, manage expiration dates, and get recipe suggestions based on your available ingredients. It uses computer vision for food detection and OpenAI for intelligent features.

## Features

- 📝 Track food items with expiration dates and fridge status
- 📸 Computer vision for food detection (apples, bananas, etc.)
- 🤚 Hand tracking for interactive features
- 🔍 Search and filter food items in/out of fridge
- 📱 Firebase integration for app integration
- 🤖 OpenAI integration for image analysis

## Prerequisites

- Python 3.8+
- Firebase account and credentials
- OpenAI API key
- OpenCV with contrib modules
- TensorFlow Lite
- MediaPipe

## Installation

1. Clone the repository:

```bash
git clone https://github.com/oorischubert/FridgeMate.git
cd FridgeMate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create a .env file with your credentials
OPENAI_API_KEY=your_openai_api_key
FIREBASE_CREDENTIALS=your_firebase_credentials.json
```

## Usage

### Basic Food Management

```python
from food import Food
from datetime import datetime, timedelta

# Create a new food item
apple = Food(
    name="Apple",
    expirationDate=datetime.now() + timedelta(days=7),
    count=5,
    inFridge=True,
    foodType="fruit",
    description="a red apple"
)

# Convert to dictionary for storage
food_dict = apple.to_dict()

# Check if expired
expired = apple.isExpired()
```

### Firebase Integration

```python
from firebase import FirebaseUploader

# Initialize Firebase
firebase = FirebaseUploader()

# Upload food item
food_id = firebase.upload_food(apple)

# Get food items
foods = firebase.get_foods(in_fridge=True, limit=100)

# Update food status
firebase.update_food_status(food_id, in_fridge=False)
```

### AI Features

```python
from openapi import OpenAPI

# Initialize OpenAI
api = OpenAPI()

# Analyze food image
analysis = api.getLabel("path/to/food.jpg", "Label prompt")
```

### Computer Vision

The system includes a detector server that can:

- Detect food items using TensorFlow Lite
- Track hand movements using MediaPipe
- Process video streams in real-time
- Send detection results via NATS messaging

## License

This project is licensed under the MIT License.

