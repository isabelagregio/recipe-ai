import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
import ast

DATASET_PATH = "data/RecipeNLG_dataset.csv"

print("🔄 Loading dataset...")
df = pd.read_csv(DATASET_PATH).dropna().reset_index(drop=True)
df = df[["title", "NER", "ingredients", "directions"]]

df["NER_list"] = df["NER"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df["NER_str"] = df["NER_list"].apply(lambda lst: " ".join(lst))

print(f"✅ Dataset Loaded: {df.shape[0]} recipes.")

vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
ingredient_vectors = vectorizer.fit_transform(df["NER_str"])

nn = NearestNeighbors(n_neighbors=1, metric="cosine")
nn.fit(ingredient_vectors)

def find_similar_recipe(user_ingredients):
    input_str = ", ".join(user_ingredients)
    cleaned_input = clean_ingredients(input_str)
    input_vector = vectorizer.transform([cleaned_input])
    distances, indices = nn.kneighbors(input_vector)

    results = []
    for idx in indices[0]:
        recipe = df.iloc[idx]
        results.append({
            "title": recipe["title"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"],
        })
    return results


def clean_ingredients(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())