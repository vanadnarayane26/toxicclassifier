import pandas as pd
from pathlib import Path

class Dataloader:
    def __init__(self, data_path:str):
        
        self.path = Path(data_path)
    
    def load_data(self):
        
        df = pd.read_csv(self.path)
        return df
    
    
    