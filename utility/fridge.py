

class Fridge:
    def __init__(self, temperature: int, humidity: int, uid: str, isOpen: bool):
        self.temperature = temperature
        self.humidity = humidity
        self.uid = uid
        self.isOpen = isOpen

    def __str__(self) -> str:
        return(f"UID: {self.uid}\nTemp: {self.temperature}\nHumidity: {self.humidity}\nIsOpen: {self.isOpen}\n")
        
    def to_dict(self) -> dict:
        return {
            'uid': self.uid,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'isOpen': self.isOpen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Fridge':
        return cls(
            uid=data['nuidame'],
            temperature=data.get('temperature',0),
            humidity=data.get('humidity', 0),
            isOpen=data.get('isOpen', False),
        )

    