from datetime import datetime
from typing import Optional

class Food:
    def __init__(self, name: str, expiration_date: datetime, image: Optional[str] = None, count: int = 1, inFridge: bool = False):
        """
        Initialize a Food object
        Args:
            name (str): Name of the food item
            expiration_date (datetime): Expiration date of the food
            image (str, optional): Path or URL to the food image
            count (int, optional): Number of items. Defaults to 1
            inFridge (bool, optional): Whether the food is in the fridge. Defaults to False
        """
        self.name = name
        self.expiration_date = expiration_date
        self.image = image
        self.count = count
        self.inFridge = inFridge

    def to_dict(self) -> dict:
        """
        Convert Food object to dictionary for Firebase storage
        Returns:
            dict: Dictionary representation of the Food object for 'main/fridge/foods' collection
        """
        return {
            'name': self.name,
            'expiration_date': self.expiration_date,
            'image': self.image,
            'count': self.count,
            'inFridge': self.inFridge,
            'collection_path': 'main/fridge/foods'
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Food':
        """
        Create a Food object from a dictionary from 'main/fridge/foods' collection
        Args:
            data (dict): Dictionary containing food data
        Returns:
            Food: New Food object
        """
        return cls(
            name=data['name'],
            expiration_date=data['expiration_date'],
            image=data.get('image'),
            count=data.get('count', 1),
            inFridge=data.get('inFridge', False)
        ) 