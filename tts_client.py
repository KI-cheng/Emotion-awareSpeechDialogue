import requests
import wave
import pyaudio
import sounddevice as sd
import soundfile as sf


def send_request(url: str, payload: dict):
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            with open('./result_wav/result.wav', 'wb') as f:
                f.write(response.content)
            f.close()
            return True
        elif response.status_code == 422:
            errors = response.json()
            for error in errors:
                print(f"Error at location {error['loc']}: {error['msg']}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None


# def play_wav(filenpath):
#     wf = wave.open(filenpath, "rb")
#     p = pyaudio.PyAudio()
#
#     # 打开音频流
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)
#     chunk = 1024
#     wav_data = wf.readframes(chunk)
#     while wav_data:
#         stream.write(wav_data)
#         wav_data = wf.readframes(chunk)
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
def play_wav(filenpath="./result_wav/result.wav"):
    data, samplerate = sf.read(filenpath)
    sd.play(data, samplerate)
    sd.wait()  # Wait until audio finishes playing

def request_setting(text="今天的天气无敌好！"):
    api_url = "http://127.0.0.1:8080/v1/tts"
    data = {
        "text": text,
        "chunk_length": 1000,
        "format": "wav",
        "reference_id": "audio_1",
        "use_memory_cache": "on",
        "streaming": False,
        "normalize": False,
        "max_new_tokens": 2048,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "temperature": 0.8
    }

    if send_request(api_url, data):
        play_wav(filenpath="./result_wav/result.wav")


if __name__ == "__main__":
    # text = """Our project introduces it."""
    # request_setting(text)
    play_wav(filenpath="./result_wav/result.wav")