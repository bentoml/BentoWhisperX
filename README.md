<div align="center">
    <h1 align="center">Serving WhisperX with BentoML</h1>
</div>

[WhisperX](https://github.com/m-bain/WhisperX) provides fast automatic speech recognition with word-level timestamps and speaker diarization.

This is a BentoML example project, demonstrating how to build a speech recognition inference API server, using the WhisperX project. See [here](https://github.com/bentoml/BentoML?tab=readme-ov-file#%EF%B8%8F-what-you-can-build-with-bentoml) for a full list of BentoML example projects.

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.
- If you want to test the project locally, install FFmpeg on your system.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoWhisperX.git
cd BentoWhisperX
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

For detailed explanations of the Service code, see [WhisperX: Speech recognition](https://docs.bentoml.org/en/latest/use-cases/audio/whisperx.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html) and set your Hugging Face access token in `bentofile.yaml`, then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
