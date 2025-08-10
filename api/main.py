from fastapi import FastAPI, Query
from typing import List

from app.model import find_similar_recipe

app = FastAPI(title="Recipe Search API", version="1.0")

@app.get("/search")
def search_recipes(ingredients: List[str] = Query(..., description="Lista de ingredientes detectados")):
    """
    Busca receitas semelhantes no RecipeNLG com base nos ingredientes fornecidos.
    Exemplo: /search?ingredients=milk&ingredients=vanilla
    """
    results = find_similar_recipe(ingredients)
    return {"query": ingredients, "results": results}
