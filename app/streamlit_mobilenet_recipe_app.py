import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

MODEL_PATH = "../model/mobilenet_recipe_ai_ingredients.h5"
DATASET_PATH = "../data/RecipeNLG_dataset.csv"
CLASS_NAMES = ['banana', 'bread', 'carrot', 'cheese', 'chicken-meat', 'chocolate',
               'egg', 'flour', 'lemon', 'milk', 'onion', 'pineapple', 'potato', 'rice', 'tomato']

@st.cache_resource
def load_classification_model(path=MODEL_PATH):
    model = tf.keras.models.load_model(path)
    return model

@st.cache_resource
def load_recipe_data(path=DATASET_PATH):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    df["NER_list"] = df["NER"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["NER_list_str"] = df["NER_list"].apply(lambda lst: str(lst))
    df["NER_str"] = df["NER_list"].apply(lambda lst: " ".join(lst))
    return df

@st.cache_resource
def prepare_search_model(df):
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df["NER_str"])
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(vectors)
    return vectorizer, nn

def preprocess_image(pil_image, target_size=(224, 224)):
    img = pil_image.convert('RGB').resize(target_size)
    x = np.array(img).astype(np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def predict(image_data, model, top_k=3):
    x = preprocess_image(image_data)
    preds = model.predict(x)[0]

    if 0.999 <= preds.sum() <= 1.001:
        probs = preds
    else:
        exp = np.exp(preds - np.max(preds))
        probs = exp / exp.sum()

    idxs = np.argsort(probs)[::-1][:top_k]
    results = [(CLASS_NAMES[i], float(probs[i]), int(i)) for i in idxs]
    return results, probs

def find_similar_recipes(ingredients_list, df, vectorizer, nn):
    input_str = " ".join(ingredients_list)
    input_vector = vectorizer.transform([input_str])
    distances, indices = nn.kneighbors(input_vector)

    recipes = []
    for idx in indices[0]:
        recipe = df.iloc[idx]
        recipes.append({
            "title": recipe["title"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"]
        })
    return recipes

# --- Streamlit UI ---
st.set_page_config(page_title="Recipe AI", layout="wide")
st.title("🥗 Recipe AI")

col1, spacer, col2 = st.columns([1, 0.2, 2])


with col1:
    st.header("📸 Upload ingredient photo")
    uploaded_file = st.file_uploader("Send an image (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_data = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image_data, caption="Image sent", use_container_width=False, width=250)

        if st.button("🔍 Identify ingredient"):
            with st.spinner("⏳ Loading classification model..."):
                model = load_classification_model(MODEL_PATH)

            with st.spinner("⏳ Predicting..."):
                results, probs = predict(image_data, model, top_k=3)

            top_label, top_prob, _ = results[0]

            st.markdown("### 🍽 Detected Ingredient")
            st.markdown(f"**{top_label.capitalize()}**")

            with st.spinner("⏳ Loading recipes and search model..."):
                df = load_recipe_data(DATASET_PATH)
                vectorizer, nn = prepare_search_model(df)

            with st.spinner("⏳ Searching for recipes..."):
                recipes = find_similar_recipes([top_label], df, vectorizer, nn)

            st.session_state["recipes"] = recipes
            st.session_state["detected"] = (top_label, top_prob)

    else:
        st.info("Send an image to start.")

with col2:
    st.header("🍽️ Suggested Recipes")

    recipes = st.session_state.get("recipes", None)

    if recipes:
        for r in recipes:
            st.markdown(f"### {r['title']}")

            try:
                ingredients_list = ast.literal_eval(r['ingredients'])
            except Exception:
                ingredients_list = [r['ingredients']]

            try:
                directions_list = ast.literal_eval(r['directions'])
            except Exception:
                directions_list = [r['directions']]

            ing_col, dir_col = st.columns(2)

            with ing_col:
                st.markdown('<p style="font-size:22px; font-weight:bold;">🛒 Ingredients:</p>', unsafe_allow_html=True)
                for i, ing in enumerate(ingredients_list, 1):
                    st.markdown(f'<p style="font-size:18px;">{i}. {ing}</p>', unsafe_allow_html=True)

            with dir_col:
                st.markdown('<p style="font-size:22px; font-weight:bold;">📖 Directions:</p>', unsafe_allow_html=True)
                for i, step in enumerate(directions_list, 1):
                    st.markdown(f'<p style="font-size:18px;">{i}. {step.strip()}</p>', unsafe_allow_html=True)

        st.markdown("---")
    else:
        st.info("Recipes will appear here after classification.")