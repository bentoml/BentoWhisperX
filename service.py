import bentoml

from pathlib import Path

LANGUAGE_CODE = "en"


@bentoml.service(traffic={"timeout": 30})
class BentoWhisperX:    

    def __init__(self):
        import torch
        import whisperx

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.model = whisperx.load_model("large-v2", self.device, compute_type=compute_type, language=LANGUAGE_CODE)
        self.model_a, self.metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=self.device)

    @bentoml.api
    def transcribe(self, audio_file: Path) -> list:
        import whisperx

        batch_size = 16 # reduce if low on GPU mem
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=batch_size)
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

        return result["segments"]
