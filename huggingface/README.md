
---
language: en
license: other
license_name: unspecified
library_name: keras
tags:
  - tensorflow
  - keras
  - mobilenetv2
  - computer-vision
  - image-classification
  - food
  - ingredients
  - recipe-recommendation
model-index:
  - name: Recipe AI Ingredient Classifier
    results: []
---

# Recipe AI — Ingredient Classifier

## Model Description

This model classifies food ingredient images into 15 categories using MobileNetV2 and TensorFlow/Keras.

It was developed as the computer vision component of [Recipe AI](https://github.com/isabelagregio/recipe-ai), an end-to-end application that identifies food ingredients from images and recommends recipes using TF-IDF and nearest-neighbor retrieval.

## Architecture

- Backbone: MobileNetV2
- Framework: TensorFlow / Keras
- Task: Single-label image classification
- Input: RGB image, resized to 224 × 224 pixels
- Output: 15 classification scores

## Supported Classes

The output indices correspond to these classes, in order:

```python
CLASS_NAMES = [
    "banana", "bread", "carrot", "cheese",
    "chicken-meat", "chocolate", "egg", "flour",
    "lemon", "milk", "onion", "pineapple",
    "potato", "rice", "tomato"
]
```

## Preprocessing

Images must be converted to RGB, resized to 224 × 224, and processed using:

```python
tf.keras.applications.mobilenet_v2.preprocess_input
```

## Usage

```python
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "isabelagregio/recipe-ai-ingredient-classifier"

CLASS_NAMES = [
    "banana", "bread", "carrot", "cheese",
    "chicken-meat", "chocolate", "egg", "flour",
    "lemon", "milk", "onion", "pineapple",
    "potato", "rice", "tomato"
]

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="mobilenet_recipe_ai_ingredients.keras"
)

model = tf.keras.models.load_model(model_path)

image = Image.open("ingredient.jpg").convert("RGB")
image = image.resize((224, 224))

x = np.asarray(image, dtype=np.float32)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = np.expand_dims(x, axis=0)

scores = model.predict(x, verbose=0)[0]

if not np.isclose(scores.sum(), 1.0, atol=1e-3):
    scores = tf.nn.softmax(scores).numpy()

prediction = CLASS_NAMES[int(np.argmax(scores))]
print(prediction)
```

## Evaluation

The original project reports a classification accuracy of **90.57%**.

The exact evaluation split, sample count, and additional class-level metrics should be documented before this value is interpreted as a reproducible benchmark.

## Project

- [GitHub Repository](https://github.com/isabelagregio/recipe-ai)
