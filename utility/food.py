from datetime import datetime
from typing import Optional

FOOD_TYPES = ['fruits','vegetables','meats',"grains","dairy","drinks"]
class Food:
    def __init__(self, name: str, description: str, foodType: str, expirationDate: datetime, count: int = 1, inFridge: bool = False):
        self.name = name
        self.expirationDate = expirationDate
        self.count = count
        self.inFridge = inFridge
        self.description = description
        self.foodType = foodType
        

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'expirationDate': self.expirationDate,
            'count': self.count,
            'inFridge': self.inFridge,
            'description': self.description,
            'foodType': self.foodType
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Food':
        return cls(
            name=data['name'],
            expirationDate=data['expirationDate'],
            count=data.get('count', 1),
            inFridge=data.get('inFridge', False),
            description=data.get("description",'none'),
            foodType=data.get("foodType","none")
        )