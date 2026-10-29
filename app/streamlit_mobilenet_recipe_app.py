import ast
import html
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Recipe AI | Your kitchen, reimagined",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
HF_REPO_ID = "isabelagregio/recipe-ai-ingredient-classifier"
MODEL_FILENAME = "mobilenet_recipe_ai_ingredients.keras"
MODEL_PATH = BASE_DIR.parent / "model" / MODEL_FILENAME
DATA_PATH = BASE_DIR.parent / "data" / "Recipe_dataset.csv"

CLASS_NAMES = [
    "banana", "bread", "carrot", "cheese", "chicken-meat", "chocolate",
    "egg", "flour", "lemon", "milk", "onion", "pineapple", "potato",
    "rice", "tomato",
]
MAX_UPLOADS = 5

LABELS = {
    "chicken-meat": "Chicken",
    "egg": "Egg",
}


def pretty_label(label):
    return LABELS.get(label, label.replace("-", " ").title())


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
      --cream: #fbfaf6;
      --ink: #22302a;
      --muted: #6b756d;
      --orange: #ec7848;
      --green: #426b53;
      --line: #e9e7df;
    }
    html, body, [class*="css"], [data-testid="stApp"] {
      font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stAppViewContainer"] { background: var(--cream); }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1170px; padding-top: 1.45rem; padding-bottom: 4rem; }
    h1, h2, h3, .outfit { font-family: 'Outfit', sans-serif !important; }
    .brand { color: var(--ink); font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 800; letter-spacing: -.045em; }
    .brand-dot { color: var(--orange); }
    .top-note { color: var(--muted); font-size: .82rem; text-align: right; padding-top: .35rem; }
    .eyebrow { color: var(--green); font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    .hero-title { font-family: 'Outfit', sans-serif; font-size: clamp(2.8rem, 5.7vw, 5.25rem); line-height: 1.04; font-weight: 800; letter-spacing: -.06em; color: var(--ink); margin: .6rem 0 .9rem; }
    .hero-title span { color: var(--orange); }
    .hero-sub { color: #69756d; font-size: 1.05rem; line-height: 1.7; max-width: 590px; margin: 0 0 1.25rem; }
    .hero-panel { background: #f1f0e8; border: 1px solid #e6e7dc; border-radius: 26px; padding: 1.8rem; min-height: 240px; display: flex; align-items: center; justify-content: center; text-align: center; position: relative; overflow: hidden; }
    .hero-panel:before { content: ''; display: block; position: absolute; width: 180px; height: 180px; right: -55px; top: -65px; border-radius: 100%; background: #e4e8d7; }
    .hero-emoji { font-size: 5.7rem; line-height: 1.25; position: relative; filter: drop-shadow(0 12px 13px rgba(65,70,44,.13)); }
    .hero-panel p { position: relative; color: #637266; font-weight: 600; margin: .7rem 0 0; font-size: .82rem; }
    .tagline-pills { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: .25rem; }
    .tagline-pills span { background: #eff1e8; color: #426b53; border: 1px solid #e1e8d9; border-radius: 999px; padding: .37rem .7rem; font-size: .77rem; font-weight: 700; }
    .section-head { font-family: 'Outfit', sans-serif; font-size: 1.72rem; letter-spacing: -.04em; font-weight: 750; color: var(--ink); margin: .45rem 0 .25rem; }
    .section-sub { color: var(--muted); font-size: .9rem; margin-bottom: .8rem; }
    .step-count { display: inline-block; font-size: .78rem; letter-spacing: 0; border: 1px solid #e7ded2; background: #fcf1e7; color: #b3633e; border-radius: 8px; padding: .22rem .5rem; vertical-align: middle; margin-right: .55rem; }
    .fineprint { color: #8a9189; font-size: .75rem; line-height: 1.55; }
    .ingredient-wrap { margin: .25rem 0 .9rem; display: flex; gap: .5rem; flex-wrap: wrap; }
    .ingredient-chip { display: inline-flex; gap: .35rem; align-items: center; color: #355e45; border-radius: 999px; border: 1px solid #cddfce; background: #e8f1e6; font-size: .85rem; padding: .43rem .8rem; font-weight: 700; }
    .image-caption { font-size: .79rem; color: #586d5e; font-weight: 700; margin-top: .4rem; }
    .score-line { color: #738478; font-size: .72rem; margin-bottom: .45rem; }
    .recipe-card { background: #fff; border: 1px solid var(--line); border-radius: 21px; padding: 1.25rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(28,44,28,.035); }
    .recipe-top { display: flex; justify-content: space-between; gap: 1rem; align-items: start; margin-bottom: .5rem; }
    .recipe-name { color: #22302a; font-family: 'Outfit', sans-serif; font-size: 1.34rem; font-weight: 750; letter-spacing: -.03em; line-height: 1.24; }
    .recipe-index { background: #fff0e7; color: #b96e46; padding: .35rem .6rem; border-radius: 999px; font-size: .72rem; font-weight: 800; white-space: nowrap; }
    .recipe-meta { color: #7f8a7f; font-size: .77rem; margin: .35rem 0 .65rem; }
    .mini-title { font-size: .78rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: #54715a; margin: .8rem 0 .3rem; }
    .recipe-list { margin: .1rem 0 .7rem; padding-left: 1.2rem; font-size: .89rem; color: #4a584e; line-height: 1.7; }
    .empty { background: #fff; border: 1px dashed #d9ddcf; border-radius: 20px; padding: 2rem 1.1rem; text-align: center; color: #7a857a; }
    .empty-icon { font-size: 2.5rem; }
    .empty-title { font-family: 'Outfit', sans-serif; color: var(--ink); font-weight: 750; font-size: 1.12rem; margin-top: .45rem; }
    .empty-copy { font-size: .88rem; margin: .45rem auto; max-width: 430px; }
    .footer-rule { border-top: 1px solid #e4e7de; margin-top: 2rem; padding-top: 1rem; color: #869087; text-align: center; font-size: .76rem; }
    div[data-testid="stFileUploader"] { background: #fff; border-radius: 20px; padding: .3rem .8rem; border: 1px solid #eceae3; }
    [data-testid="stFileUploaderDropzone"] { border-radius: 15px; background: #fbfaf6; border: 1.5px dashed #cbd5c8; }
    div.stButton > button[kind="primary"] { background: #e97949; border-color: #e97949; border-radius: 12px; color: #fff; font-weight: 750; min-height: 3.1rem; box-shadow: 0 8px 16px rgba(233,121,73,.13); }
    div.stButton > button[kind="primary"]:hover { background: #d9673b; border-color: #d9673b; color: #fff; }
    div.stButton > button[kind="secondary"] { border-radius: 11px; font-weight: 700; }
    div[data-testid="stExpander"] { border: 0; background: #f7f8f2; border-radius: 12px; }
    @media (max-width: 720px) {
      .block-container { padding: 1rem 1.1rem 2.4rem; }
      .hero-title { font-size: 2.65rem; }
      .hero-panel { min-height: 165px; padding: 1rem; }
      .hero-emoji { font-size: 4rem; }
      .top-note { text-align: left; }
      .recipe-card { padding: 1rem; }
    }
    /* Expander arrow: visible even when collapsed */
    [data-testid="stExpander"] summary svg {
        color: #294A36 !important;
        stroke: #294A36 !important;
        fill: none !important;
    }

    /* Expander label */
    [data-testid="stExpander"] summary {
        color: #294A36 !important;
        background-color: #F0F5ED !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Original ML pipeline (unchanged)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_and_load_dataset():
    df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)
    df["NER_list"] = df["NER"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["NER_list_str"] = df["NER_list"].apply(str)
    df["NER_str"] = df["NER_list"].apply(lambda items: " ".join(items))
    return df


@st.cache_resource(show_spinner=False)
def load_classification_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    except Exception as exc:
        if MODEL_PATH.is_file():
            model_path = str(MODEL_PATH)
        else:
            raise RuntimeError(
                "Could not download the model and no local fallback was found."
            ) from exc
    return tf.keras.models.load_model(model_path)


@st.cache_resource(show_spinner=False)
def prepare_search_model(_df):
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(_df["NER_str"])
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(vectors)
    return vectorizer, nn


def preprocess_image(pil_image, target_size=(224, 224)):
    img = pil_image.convert("RGB").resize(target_size)
    x = np.array(img).astype(np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return np.expand_dims(x, axis=0)


def predict(image_data, model, top_k=3):
    x = preprocess_image(image_data)
    preds = model.predict(x, verbose=0)[0]
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
    _, indices = nn.kneighbors(input_vector)
    return [
        {
            "title": df.iloc[idx]["title"],
            "ingredients": df.iloc[idx]["ingredients"],
            "directions": df.iloc[idx]["directions"],
        }
        for idx in indices[0]
    ]


# -----------------------------------------------------------------------------
# Presentation helpers
# -----------------------------------------------------------------------------
def parse_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        return [value]
    return [str(value)]


def render_list(values, ordered=False):
    tag = "ol" if ordered else "ul"
    children = "".join(f"<li>{html.escape(str(item))}</li>" for item in values)
    return f'<{tag} class="recipe-list">{children}</{tag}>'


def render_recipe(recipe, index):
    title = html.escape(str(recipe["title"]))
    ingredients = parse_list(recipe["ingredients"])
    directions = parse_list(recipe["directions"])
    st.markdown(
        f'<div class="recipe-card">'
        f'<div class="recipe-top"><div class="recipe-name">{title}</div>'
        f'<div class="recipe-index">RECIPE {index:02d}</div></div>'
        f'<div class="recipe-meta">🍴 {len(ingredients)} ingredients &nbsp; · &nbsp; AI-powered match</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    with st.expander("View ingredients & directions", expanded=False):
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            st.markdown('<div class="mini-title">🛒 Ingredients</div>', unsafe_allow_html=True)
            st.markdown(render_list(ingredients), unsafe_allow_html=True)
        with right:
            st.markdown('<div class="mini-title">👩‍🍳 Directions</div>', unsafe_allow_html=True)
            st.markdown(render_list(directions, ordered=True), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "detected_labels" not in st.session_state:
    st.session_state.detected_labels = []
if "recipes" not in st.session_state:
    st.session_state.recipes = []


# -----------------------------------------------------------------------------
# Header + hero
# -----------------------------------------------------------------------------
nav_left, nav_right = st.columns([3, 2], vertical_alignment="center")
with nav_left:
    st.markdown('<div class="brand">🍋 recipe<span class="brand-dot">.</span>ai</div>', unsafe_allow_html=True)
with nav_right:
    st.markdown('<div class="top-note">Computer vision meets your kitchen ✦</div>', unsafe_allow_html=True)

st.write("")
hero_copy, hero_visual = st.columns([1.65, 1], gap="large", vertical_alignment="center")
with hero_copy:
    st.markdown('<div class="eyebrow">THE SMARTER WAY TO COOK</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Less waste.<br>More <span>taste.</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Your ingredients have potential. Snap a few photos, '
        'let AI recognize what you have, and discover recipes worth making.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tagline-pills"><span>📸 Recognize ingredients</span>'
        '<span>🥬 Cook what you have</span><span>✨ Discover recipes</span></div>',
        unsafe_allow_html=True,
    )
with hero_visual:
    st.markdown(
        '<div class="hero-panel"><div><div class="hero-emoji">🥑🥕🍳</div>'
        '<p>From ingredients to inspiration.</p></div></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# -----------------------------------------------------------------------------
# Upload + results
# -----------------------------------------------------------------------------
left, right = st.columns([0.95, 1.25], gap="large")
with left:
    st.markdown(
        '<div class="section-head"><span class="step-count">01</span> Start with ingredients</div>'
        '<div class="section-sub">Upload up to 5 photos, ideally one ingredient per image.</div>',
        unsafe_allow_html=True,
    )

    uploads = st.file_uploader(
        "Choose ingredient photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="JPG or PNG. The classifier recognizes 15 ingredient categories.",
    )

    if len(uploads) > MAX_UPLOADS:
        st.warning(f"Please select at most {MAX_UPLOADS} images. Only the first {MAX_UPLOADS} will be used.")
        uploads = uploads[:MAX_UPLOADS]

    st.markdown('<div class="fineprint">Tip: Use clear, well-lit pictures with a single main ingredient. This model recognizes 15 food categories.</div>', unsafe_allow_html=True)
    st.write("")

    if uploads:
        st.markdown('<div class="mini-title">YOUR PHOTOS</div>', unsafe_allow_html=True)
        preview_columns = st.columns(min(len(uploads), 3))
        for i, uploaded_file in enumerate(uploads):
            with preview_columns[i % len(preview_columns)]:
                try:
                    img = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded_file.getvalue())))
                    st.image(img, use_container_width=True)
                    st.caption(uploaded_file.name[:30])
                except Exception:
                    st.caption(f"Could not preview: {uploaded_file.name}")

    identify_clicked = st.button(
        "✨ Find recipes for me",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploads),
    )

    if identify_clicked:
        try:
            with st.spinner("Recognizing your ingredients..."):
                model = load_classification_model()
                predicted = []
                for uploaded_file in uploads:
                    image = ImageOps.exif_transpose(
                        Image.open(io.BytesIO(uploaded_file.getvalue()))
                    )
                    results, _ = predict(image, model, top_k=1)
                    label, confidence, _ = results[0]
                    predicted.append({
                        "filename": uploaded_file.name,
                        "label": label,
                        "confidence": confidence,
                    })

            labels = list(dict.fromkeys(item["label"] for item in predicted))
            with st.spinner("Finding your recipe matches..."):
                df = download_and_load_dataset()
                vectorizer, nn = prepare_search_model(df)
                recipes = find_similar_recipes(labels, df, vectorizer, nn)

            # Store results only after the complete pipeline succeeds.
            st.session_state.predictions = predicted
            st.session_state.detected_labels = labels
            st.session_state.recipes = recipes
        except Exception as exc:
            st.error("Something went wrong while processing your images. Please try again.")
            with st.expander("Technical details"):
                st.code(str(exc))

    predictions = st.session_state.predictions
    if predictions:
        st.write("")
        st.markdown('<div class="section-head"><span class="step-count">02</span> What we found</div>', unsafe_allow_html=True)
        chips = "".join(
            f'<span class="ingredient-chip">✓ {html.escape(pretty_label(label))}</span>'
            for label in st.session_state.detected_labels
        )
        st.markdown(f'<div class="ingredient-wrap">{chips}</div>', unsafe_allow_html=True)
        st.markdown('<div class="mini-title">DETECTION DETAILS</div>', unsafe_allow_html=True)
        for item in predictions:
            label = pretty_label(item["label"])
            confidence = item["confidence"]
            st.markdown(
                f'<div class="image-caption">{html.escape(item["filename"])} · '
                f'{html.escape(label)}</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(confidence), text=f"Confidence: {confidence:.0%}")
        st.markdown(
            '<div class="fineprint">Confidence is the model\'s estimated probability, not a guarantee of correctness.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear results", use_container_width=False):
            for key in ("predictions", "detected_labels", "recipes"):
                st.session_state[key] = []
            st.rerun()

with right:
    st.markdown(
        '<div class="section-head"><span class="step-count">03</span> Made for your kitchen</div>'
        '<div class="section-sub">Three recipe ideas based on the ingredients identified in your photos.</div>',
        unsafe_allow_html=True,
    )
    recipes = st.session_state.recipes
    if recipes:
        st.success(f"Found {len(recipes)} recipe ideas. Pick one and get cooking!")
        for i, recipe in enumerate(recipes, start=1):
            render_recipe(recipe, i)
    else:
        st.markdown(
            '<div class="empty"><div class="empty-icon">🥘</div>'
            '<div class="empty-title">Your next favorite meal starts here.</div>'
            '<div class="empty-copy">Add your ingredient photos and tap '
            '<strong>Find recipes for me</strong>. Your recommendations will appear here.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer-rule">Made with 🧡 and machine learning · '
    'MobileNetV2 + TF-IDF + Nearest Neighbors · '
    '<a href="https://github.com/isabelagregio/recipe-ai" target="_blank">View on GitHub ↗</a></div>',
    unsafe_allow_html=True,
)
