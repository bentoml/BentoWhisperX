<div align="center">
    <h1 align="center">Serving WhisperX with BentoML</h1>
</div>

[WhisperX](https://github.com/m-bain/WhisperX) provides fast automatic speech recognition with word-level timestamps and speaker diarization.

This is a BentoML example project, demonstrating how to build a speech recognition inference API server, using the WhisperX project. See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

If you want to test the project locally, install FFmpeg on your system.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoWhisperX.git
cd BentoWhisperX

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.
Please note that you may need to request access to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), then provide your Hugging Face token when running the Service.

```python
$ HF_TOKEN=<your hf access token> bentoml serve .

2024-01-18T09:01:15+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:BentoWhisperX" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -s \
     -X POST \
     -F 'audio_file=@female.wav' \
     http://localhost:3000/transcribe
```

Python client

```python
from pathlib import Path
import bentoml

with bentoml.SyncHTTPClient('http://localhost:3000') as client:
    audio_url = 'https://example.org/female.wav'
    response = client.transcribe(audio_file=audio_url)
    print(response)
```

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html) and set your Hugging Face access token in `bentofile.yaml`, then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
