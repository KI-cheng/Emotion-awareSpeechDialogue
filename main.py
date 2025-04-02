import time

from LLMClient import LLMClient
from unsloth_loader import unsloth_Qwen
from tts_client import request_setting
from vad import record_and_process


def wav_to_text(asr):
    audio_path = record_and_process()
    res, emotion = asr.generate(audio_path=audio_path)
    return res, emotion


def main(asr, qwen, gpt):
    while True:
        first = time.time()
        # ask, emotion = wav_to_text(asr)
        # print(ask)
        # print(emotion)
        ask = "Hi, What about today's weather?"
        print("Qwen2.5_7B进行情感识别...")
        emo = qwen.identification(ask).strip()
        print(f"情感识别为{emo}!")
        print("chat-gpt-turbo-3.5推理进行回复...")
        reply = gpt.get_response(ask=ask, emo=emo)
        print(f"CyberPet:{reply}")
        print("reply转发至tts模块...")
        request_setting(text=reply)
        print(time.time()-first)


if __name__ == "__main__":
    print("ASR模型加载中...")
    from asr import SpeechRecognizer

    asr = SpeechRecognizer()
    print("ASR模型已加载...")
    print("Qwen2.5_7B加载中...")
    qwen = unsloth_Qwen()
    print("Qwen2.5_7B已加载...")
    print("gpt-turbo-3.5加载中...")
    gpt = LLMClient()
    print("gpt-turbo-3.5已加载...")
    main(asr, qwen, gpt)
