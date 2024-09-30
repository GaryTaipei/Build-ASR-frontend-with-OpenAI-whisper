import gradio as gr
import requests
from decouple import config
from ASRAudio import *

#Set up the Transformers ASR Model
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
#修改 forced_decoder_ids 成 “<|de|>” (ID 50261) 可以轉換成德語，但是輸出本文的語言是英語
# pipe.model.config.forced_decoder_ids[0][1]=50261

# def transcribe(audio):
#     sr, y = audio
#     # Convert to mono if stereo
#     if y.ndim > 1:
#         y = y.mean(axis=1)
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))
#     return pipe({"sampling_rate": sr, "raw": y})["text"] 

def request_llm(text:str):
    api_url = config("API_URL") #呼叫 FastAPI 的 API 代理呼叫 ollama API http://localhost:11434/api/generate
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral",
        "prompt": text
    }
    response = requests.post(api_url, json=data, headers=headers)
    return response.json()['response']

def main():
    asr = ASRAudio()
    demo = gr.Blocks()
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        #create a row with two columns
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Enter text", placeholder="Please input your query here.")
            with gr.Column():
                input_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    max_length=10,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        waveform_progress_color="#0066B4",
                        skip_length=2,
                        show_controls=False,
                    ),
                )
        #create a row with two buttons
        with gr.Row():
            with gr.Column():
                response = gr.Button("Generate response", variant="primary")
            with gr.Column():
                # clear = gr.Button("Clear") #如果是作為 ClearButton ，不可以宣告為 Button，否則 gr.ClearButton.add() 會出現錯誤 
                clear = gr.ClearButton( value="Clear")
        #create a row for response
        with gr.Row():
            out = gr.Textbox(label="Response", placeholder="Response will appear here.")
            output_audio = gr.Audio(
                label="Output Audio",
                value=None,
                format="mp3",
                autoplay=True,
                streaming=False,
                interactive=False,
                show_label=False,
                visible=False,
                waveform_options=gr.WaveformOptions(
                    sample_rate=24000,
                ),
            )
        response.click(fn=lambda audio: asr.transcribe(audio[1]), inputs=[input_audio], outputs=[text]).then(request_llm,[text], [out]).then(asr.generate_audio,[out],[output_audio])
        # input_audio.input(fn=transcribe,inputs=[input_audio],outputs=[out])
        gr.ClearButton.add(clear, components=[text, input_audio, out])
    demo.queue().launch(share=False, debug=True)


if __name__ == '__main__':
    main()