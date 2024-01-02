import bentoml
import os

from pathlib import Path

LANGUAGE_CODE = "en"


@bentoml.service(traffic={"timeout": 30})
class BentoWhisperX:
    """
    This class is inspired by the implementation shown in the whisperX project.
    Source: https://github.com/m-bain/whisperX
    """

    def __init__(self):
        import torch
        import whisperx

        self.batch_size = 16 # reduce if low on GPU mem
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.model = whisperx.load_model("large-v2", self.device, compute_type=compute_type, language=LANGUAGE_CODE)
        self.model_a, self.metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=self.device)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=self.device)

    @bentoml.api
    def transcribe(self, audio_file: Path) -> list:
        import whisperx

        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

        return result
    
    @bentoml.api
    def diarize(self, audio_file: Path) -> list:
        import whisperx

        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

        diarize_segments = self.diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return result
