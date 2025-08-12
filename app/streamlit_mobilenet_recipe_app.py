import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'mobilenet_recipe_ai_ingredients.keras')

KAGGLE_DATASET = 'paultimothymooney/recipenlg'  
DATA_FILE = 'RecipeNLG_dataset.csv' 
LOCAL_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', DATA_FILE)


CLASS_NAMES = ['banana', 'bread', 'carrot', 'cheese', 'chicken-meat', 'chocolate',
               'egg', 'flour', 'lemon', 'milk', 'onion', 'pineapple', 'potato', 'rice', 'tomato']


@st.cache_resource(show_spinner=False)
def download_and_load_dataset():
    import os
    import pandas as pd
    import ast

    file_path = os.path.join(BASE_DIR, "..", "data", "Recipe_dataset.csv")

    df = pd.read_csv(file_path)

    df = df.dropna().reset_index(drop=True)

    df["NER_list"] = df["NER"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["NER_list_str"] = df["NER_list"].apply(lambda lst: str(lst))
    df["NER_str"] = df["NER_list"].apply(lambda lst: " ".join(lst))

    return df

@st.cache_resource
def load_classification_model(path=MODEL_PATH):
    model = tf.keras.models.load_model(path)
    return model

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
st.markdown("<h1 style='color:#008DF5;'>🥗 Recipe AI</h1>", unsafe_allow_html=True)

col1, spacer, col2 = st.columns([1, 0.2, 2])

with col1:
    st.markdown("<h2 style='color:#008DF5;'>📸 Upload ingredient photos</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Send up to 5 images (jpg, png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠ You can upload a maximum of 5 images.")
        uploaded_files = uploaded_files[:5]

    if uploaded_files and st.button("🔍 Identify ingredients"):
        with st.spinner("⏳ Loading classification model..."):
            model = load_classification_model(MODEL_PATH)

        detected_labels = []
        for uploaded_file in uploaded_files:
            image_data = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(image_data, caption=f"Uploaded: {uploaded_file.name}", width=200)
            results, probs = predict(image_data, model, top_k=1)
            top_label, top_prob, _ = results[0]
            detected_labels.append(top_label)

        st.session_state["detected_labels"] = detected_labels

        with st.spinner("⏳ Loading recipes and search model..."):
            df = download_and_load_dataset()
            if df is None:
                st.error("Could not load dataset.")
                st.stop()

            vectorizer, nn = prepare_search_model(df)

        with st.spinner("⏳ Searching for recipes..."):
            recipes = find_similar_recipes(detected_labels, df, vectorizer, nn)

        st.session_state["recipes"] = recipes

with col2:
    detected_labels = st.session_state.get("detected_labels", [])
    if detected_labels:
        st.markdown("<h2 style='color:#008DF5;'>🛒 Detected Ingredients</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='font-size:28px;'>{', '.join([label.capitalize() for label in detected_labels])}</h2>",
            unsafe_allow_html=True
        )

    st.markdown("<h2 style='color:#008DF5;'>🍽️ Suggested Recipes</h2>", unsafe_allow_html=True)
    recipes = st.session_state.get("recipes", None)

    if recipes:
        for r in recipes:
            st.markdown(f"<h3 style='color:#56B1F5;'>{r['title']}</h3>", unsafe_allow_html=True)
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
                st.markdown('<p style="font-size:22px; font-weight:bold; color:#56B1F5;">🛒 Ingredients:</p>', unsafe_allow_html=True)
                for ing in ingredients_list:
                    st.markdown(f'<p style="font-size:20px;">&#8226; {ing}</p>', unsafe_allow_html=True)  # &#8226; é •
            with dir_col:
                st.markdown('<p style="font-size:22px; font-weight:bold; color:#56B1F5;">📖 Directions:</p>', unsafe_allow_html=True)
                for step in directions_list:
                    st.markdown(f'<p style="font-size:20px;">&#8226; {step.strip()}</p>', unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("Recipes will appear here after sending ingredient photos.")
