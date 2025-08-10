from typing import List
from pydantic import BaseModel

class RecipeQuery(BaseModel):
    ingredients: List[str]

class RecipeResult(BaseModel):
    title: str
    ingredients: str
    directions: str
    NER: str
