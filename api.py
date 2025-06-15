
import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
from pydub import AudioSegment

import numpy as np
import librosa
from flask import Flask, request, Response,send_from_directory
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import ffmpeg

from flask_cors import CORS
from flask import make_response

import shutil

import json

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz')

default_voices = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

spk_new = []

for name in os.listdir(f"{ROOT_DIR}/voices/"):
    print(name.replace(".py",""))
    spk_new.append(name.replace(".py",""))

print("默认音色",cosyvoice.list_avaliable_spks())
print("自定义音色",spk_new)

app = Flask(__name__)

CORS(app, cors_allowed_origins="*")

CORS(app, supports_credentials=True)


def download_and_convert(mp3_url, wav_filename):
    """Downloads an MP3 file and converts it to WAV.

    Args:
        mp3_url: The URL of the MP3 file.
        wav_filename: The desired filename for the WAV file (including .wav extension).
    """
    try:
        response = requests.get(mp3_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open("temp.mp3", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Convert MP3 to WAV using pydub
        sound = AudioSegment.from_mp3("temp.mp3")
        sound.export(wav_filename, format="wav")

        print(f"音频已成功下载并转换为 {wav_filename}")

    except requests.exceptions.RequestException as e:
        print(f"下载音频时出错: {e}")
    except Exception as e:
        print(f"转换音频时出错: {e}")
    finally:
        # Clean up the temporary MP3 file
        import os
        try:
            os.remove("temp.mp3")
        except OSError as e:
            print(f"删除临时文件时出错: {e}")

def postprocess(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8, target_sr=22050): # Added target_sr to make it configurable if needed
    # Assuming librosa is available or will be imported. If not, this needs to be handled.
    # For now, let's assume librosa can be imported.
    import librosa
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if torch.is_tensor(speech) and speech.abs().max() > max_val: # Check if speech is a tensor
        speech = speech / speech.abs().max() * max_val
    elif not torch.is_tensor(speech) and np.abs(speech).max() > max_val: # Handle numpy array case
         speech = speech / np.abs(speech).max() * max_val
    # Ensure speech is a tensor before concatenating
    if not torch.is_tensor(speech):
        speech = torch.tensor(speech, dtype=torch.float32)
    if speech.ndim == 1: # Ensure it's 2D for concat
        speech = speech.unsqueeze(0)
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

@app.route("/", methods=['POST'])
def sft_post():
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = question_data.get('speaker')
    streaming = question_data.get('streaming',0)

    speed = request.args.get('speed',1.0)
    speed = float(speed)
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        buffer = io.BytesIO()

        tts_speeches = []

        for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=False,speed=speed,new_dropdown="无")):
            # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
            tts_speeches.append(j['tts_speech'])
        
        audio_data = torch.concat(tts_speeches, dim=1)
        torchaudio.save(buffer,audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        def generate():

            for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=True,speed=speed,new_dropdown="无")):

                tts_speeches = []
                buffer = io.BytesIO()
                tts_speeches.append(j['tts_speech'])
                audio_data = torch.concat(tts_speeches, dim=1)
                torchaudio.save(buffer,audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers['Content-Type'] = 'audio/ogg'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
        return response

@app.route("/save_voice", methods=['GET'])
def save_voice():

    text = request.args.get('text')
    audio = request.args.get('audio')
    voice_name = request.args.get('voice_name')


    download_and_convert(audio,"zero_test.wav")

    prompt_speech_16k = load_wav('zero_test.wav', 16000)
    tts_speeches = []
    for i, j in enumerate(cosyvoice.inference_zero_shot(text,text, prompt_speech_16k, stream=False)):
        # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
        tts_speeches.append(j['tts_speech'])
    
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save('zero_shot.wav',audio_data, 22050, format="wav")

    shutil.copyfile(f"{ROOT_DIR}/output.pt",f"{ROOT_DIR}/voices/{voice_name}.pt")

    response = app.response_class(
        response=json.dumps({"voice_name":voice_name}),
        status=200,
        mimetype='application/json'
    )
    return response





@app.route("/", methods=['GET'])
def sft_get():

    text = request.args.get('text')
    speaker = request.args.get('speaker')
    new = request.args.get('new',0)
    streaming = request.args.get('streaming',0)
    speed = request.args.get('speed',1.0)
    speed = float(speed)

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        buffer = io.BytesIO()

        tts_speeches = []

        for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=False,speed=speed,new_dropdown="无")):
            # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
            tts_speeches.append(j['tts_speech'])
        
        audio_data = torch.concat(tts_speeches, dim=1)
        torchaudio.save(buffer,audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        
        def generate():

            for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=True,speed=speed,new_dropdown="无")):

                tts_speeches = []
                buffer = io.BytesIO()
                tts_speeches.append(j['tts_speech'])
                audio_data = torch.concat(tts_speeches, dim=1)
                torchaudio.save(buffer,audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers['Content-Type'] = 'audio/ogg'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
        return response
        
        # return Response(generate(), mimetype='audio/x-wav')

                





@app.route("/tts_to_audio/", methods=['POST'])
def tts_to_audio():

    import speaker_config
    
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = speaker_config.speaker

    speed = speaker_config.speed
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    buffer = io.BytesIO()

    tts_speeches = []

    for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=False,speed=speed,new_dropdown="无")):
        # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
        tts_speeches.append(j['tts_speech'])
    
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save(buffer,audio_data, 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")



@app.route("/speakers", methods=['GET'])
def speakers():

    voices = []

    for x in default_voices:
        voices.append({"name":x,"voice_id":x})

    for name in os.listdir("voices"):
        name = name.replace(".pt","")
        voices.append({"name":name,"voice_id":name})

    response = app.response_class(
        response=json.dumps(voices),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/speakers_list", methods=['GET'])
def speakers_list():

    response = app.response_class(
        response=json.dumps(["female_calm","female","male"]),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/file/<filename>')
def uploaded_file(filename):
    return send_from_directory("音频输出", filename)


@app.route("/zero_shot_inference", methods=['POST'])
def zero_shot_inference():
    """
    Synthesizes speech from text using a prompt audio for zero-shot voice cloning.

    This endpoint accepts POST requests with `multipart/form-data`.

    Required Form Fields:
    - `text` (string): The text to be synthesized into speech.
    - `prompt_text` (string): The transcript of the prompt audio. This should closely match the content of the prompt audio.

    Audio Input (provide one of the following, `prompt_audio_file` is prioritized):
    - `prompt_audio_file` (file): An audio file (e.g., WAV, MP3) uploaded by the client. This is the preferred method for providing the prompt audio.
    - `prompt_audio_url` (string): A URL pointing to an audio file. This is used as a fallback if `prompt_audio_file` is not provided.

    Optional Query Parameter:
    - `speed` (float): Controls the speed of the synthesized speech. Defaults to 1.0. Values less than 1.0 will slow down the speech, and values greater than 1.0 will speed it up.

    Successful Response:
    - HTTP 200 OK: Returns a WAV audio file (`audio/wav`) containing the synthesized speech.

    Error Responses:
    - HTTP 400 Bad Request:
        - If required form fields (`text`, `prompt_text`, and one of `prompt_audio_file` or `prompt_audio_url`) are missing.
        - If the `speed` parameter is not a valid float.
        - If an uploaded `prompt_audio_file` cannot be converted to WAV format (e.g., unsupported audio type or corrupted file).
    - HTTP 500 Internal Server Error:
        - If downloading or converting audio from `prompt_audio_url` fails.
        - If processing the prompt audio (loading, postprocessing) fails.
        - If the core zero-shot inference process fails.
        - If the inference process returns no audio data.

    Example Usage (curl):

    1. Using File Upload:
    ```curl
    curl -X POST -F "text=Hello world, this is a test." \
         -F "prompt_text=This is the content of my prompt audio." \
         -F "prompt_audio_file=@/path/to/your/sample_audio.wav" \
         "http://127.0.0.1:9880/zero_shot_inference?speed=1.1" \
         -o output_speech.wav
    ```

    2. Using URL for Prompt Audio:
    ```curl
    curl -X POST -F "text=Another example sentence for synthesis." \
         -F "prompt_text=This is what the URL audio says." \
         -F "prompt_audio_url=http://example.com/audio_prompt.mp3" \
         "http://127.0.0.1:9880/zero_shot_inference?speed=0.9" \
         -o output_speech_from_url.wav
    ```
    """
    text = request.form.get('text')
    prompt_text = request.form.get('prompt_text')
    prompt_audio_url = request.form.get('prompt_audio_url')
    prompt_audio_file = request.files.get('prompt_audio_file')
    speed = request.args.get('speed', 1.0) # Speed can remain a query param

    if not all([text, prompt_text]) or not (prompt_audio_file or prompt_audio_url):
        return {"error": "Missing required form fields: text, prompt_text, and either prompt_audio_file or prompt_audio_url"}, 400

    try:
        speed = float(speed)
    except ValueError:
        return {"error": "Invalid speed parameter, must be a float"}, 400

    prompt_audio_target_filename = "prompt_audio.wav"
    uploaded_temp_audio_path = "uploaded_prompt_audio" # Temporary path for any uploaded file

    if prompt_audio_file:
        prompt_audio_file.save(uploaded_temp_audio_path)
        original_filename = prompt_audio_file.filename

        if original_filename.lower().endswith('.wav'):
            # If already WAV, just rename/move
            if os.path.exists(prompt_audio_target_filename):
                 os.remove(prompt_audio_target_filename) # remove previous one if any
            os.rename(uploaded_temp_audio_path, prompt_audio_target_filename)
        else:
            # Convert to WAV
            try:
                sound = AudioSegment.from_file(uploaded_temp_audio_path) # pydub infers format
                sound.export(prompt_audio_target_filename, format="wav")
                print(f"Uploaded file {original_filename} converted to {prompt_audio_target_filename}")
            except Exception as e:
                if os.path.exists(uploaded_temp_audio_path): # Clean up temp uploaded file
                    os.remove(uploaded_temp_audio_path)
                print(f"Error converting uploaded audio: {e}")
                return {"error": f"Could not convert uploaded audio file. Ensure it's a valid audio format (e.g., MP3, WAV). Error: {e}"}, 400
            finally:
                # Clean up the intermediate uploaded file if it exists and is different from target
                if os.path.exists(uploaded_temp_audio_path) and uploaded_temp_audio_path != prompt_audio_target_filename :
                    os.remove(uploaded_temp_audio_path)

    elif prompt_audio_url:
        try:
            download_and_convert(prompt_audio_url, prompt_audio_target_filename)
        except Exception as e: # download_and_convert might raise various exceptions
            print(f"Error downloading or converting audio from URL: {e}")
            return {"error": f"Failed to download or convert audio from URL: {e}"}, 500
    else:
        # This case should be caught by the initial check, but as a safeguard:
        return {"error": "No audio source provided (file or URL)."}, 400


    prompt_sr = 16000
    try:
        # Load and preprocess the prompt audio
        prompt_audio_data = load_wav(prompt_audio_target_filename, sr=prompt_sr)
        prompt_speech_16k = postprocess(prompt_audio_data)
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing prompt audio {prompt_audio_target_filename}: {e}")
        # Consider removing the temp file if it exists
        if os.path.exists(prompt_audio_target_filename):
            os.remove(prompt_audio_target_filename)
        return {"error": f"Failed to process prompt audio: {e}"}, 500


    tts_speeches = []
    try:
        for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False, speed=speed)):
            tts_speeches.append(j['tts_speech'])
    except Exception as e:
        # Log the error for debugging
        print(f"Error during zero-shot inference: {e}")
        # Consider removing the temp file if it exists
        if os.path.exists("prompt_audio.wav"):
            os.remove("prompt_audio.wav")
        return {"error": f"Inference failed: {e}"}, 500


    if not tts_speeches:
        # Clean up prompt_audio.wav if it exists
        if os.path.exists("prompt_audio.wav"):
            os.remove("prompt_audio.wav")
        return {"error": "Inference returned no audio"}, 500

    audio_data = torch.concat(tts_speeches, dim=1)

    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_data, 22050, format="wav")
    buffer.seek(0)

    # Clean up prompt_audio.wav if it exists
    if os.path.exists("prompt_audio.wav"):
        os.remove("prompt_audio.wav")

    return Response(buffer.read(), mimetype="audio/wav")
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)
