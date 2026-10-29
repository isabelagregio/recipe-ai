
from pathlib import Path

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parents[1]

LOCAL_PATH = (
    ROOT_DIR
    / "model"
    / "mobilenet_recipe_ai_ingredients.keras"
)

REPO_ID = "isabelagregio/recipe-ai-ingredient-classifier"

remote_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="mobilenet_recipe_ai_ingredients.keras",
)

local_model = tf.keras.models.load_model(LOCAL_PATH)
remote_model = tf.keras.models.load_model(remote_path)

assert local_model.input_shape == remote_model.input_shape
assert local_model.output_shape == remote_model.output_shape

# Deterministic synthetic input for equivalence testing.
# This is not an accuracy evaluation.
rng = np.random.default_rng(42)

x = rng.uniform(
    -1.0, 1.0,
    size=(2, 224, 224, 3),
).astype(np.float32)

local_predictions = local_model.predict(x, verbose=0)
remote_predictions = remote_model.predict(x, verbose=0)

np.testing.assert_allclose(
    local_predictions,
    remote_predictions,
    rtol=1e-5,
    atol=1e-6,
)

print("Model verification passed.")
print("Local and Hugging Face outputs match.")
