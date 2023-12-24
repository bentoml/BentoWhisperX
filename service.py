import bentoml

from pathlib import Path

LANGUAGE_CODE = "en"

whisper_model = bentoml.models.get("whisper-large-v2")


@bentoml.service(
    traffic={
        "timeout": 1,
    },
    resources={
        "gpu_type": "nvidia_tesla_t4",
        "gpu": 1,
    },
)
class BentoWhisperX:    

    def __init__(self):
        import torch
        import whisperx

        self.cuda_available = torch.cuda.is_available()
        device = "cuda" if self.cuda_available else "cpu"
        compute_type = "float16" if self.cuda_available else "int8"
        self.model = whisperx.load_model(whisper_model.path, device, compute_type=compute_type, language=LANGUAGE_CODE)
        self.model_a, self.metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=device)

    @bentoml.api
    async def transcribe(self, audio_file: Path) -> list:
        import whisperx

        device = "cuda" if self.cuda_available else "cpu"
        batch_size = 16 # reduce if low on GPU mem
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=batch_size)
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, device, return_char_alignments=False)

        return result["segments"]
