import bentoml

from faster_whisper.utils import download_model


if __name__ == "__main__":
    with bentoml.models.create("whisper-large-v2") as model:
        download_model("large-v2", model.path)
        print(f"Saved {model.info.tag} model under {model.path}")
