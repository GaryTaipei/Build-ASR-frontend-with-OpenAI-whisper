from openai import OpenAI
from decouple import config
import tempfile
from transformers import pipeline
import numpy as np

class ASRAudio():
    """
    ASR(Automatic Speech Recognition) 類別，用於轉換音訊成文字及文字轉換成音訊
    """
    def __init__(self):
        self.sr = 24000 #定義 sampling rate 預設為 24000
        self.pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
        #修改 forced_decoder_ids 成 “<|de|>” (ID 50261) 可以轉換成德語，但是輸出本文的語言是英語
        # pipe.model.config.forced_decoder_ids[0][1]=50261
        
    def transcribe(self,audio):
        self.y = audio
        # Convert to mono if stereo
        if self.y.ndim > 1:
            self.y = self.y.mean(axis=1)
        self.y = self.y.astype(np.float32)
        self.y /= np.max(np.abs(self.y))
        return self.pipe({"sampling_rate": self.sr, "raw": self.y})["text"] 

    def generate_audio(self,text:str):
        client = OpenAI(
            api_key=config("OPENAI_API_KEY"),
        )
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3",
        ) as response:
            # response.stream_to_file("speech.mp3")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(response.read())
            temp_file_path = temp_file.name
            return temp_file_path
        # response = client.audio.speech.create(
        #     model="tts-1",
        #     voice="shimmer",
        #     input=text,
        # )
        # response.stream_to_file("output.mp3")

