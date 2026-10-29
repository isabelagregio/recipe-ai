
from pathlib import Path

from huggingface_hub import HfApi

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "model"
    / "mobilenet_recipe_ai_ingredients.keras"
)

MODEL_CARD_PATH = ROOT_DIR / "huggingface" / "README.md"

REPO_ID = "isabelagregio/recipe-ai-ingredient-classifier"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not MODEL_CARD_PATH.is_file():
        raise FileNotFoundError(
            f"Model Card not found: {MODEL_CARD_PATH}"
        )

    api = HfApi()

    print("Creating or accessing Hugging Face repository...")

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    print("Uploading model weights...")

    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo=MODEL_PATH.name,
        repo_id=REPO_ID,
        repo_type="model",
    )

    print("Uploading Model Card...")

    api.upload_file(
        path_or_fileobj=str(MODEL_CARD_PATH),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(
        "Model published successfully:\n"
        f"https://huggingface.co/{REPO_ID}"
    )


if __name__ == "__main__":
    main()
