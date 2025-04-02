from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

SENSE_VOICE_PATH = "E:\\ModelsFiles\\modelscope\\models\\iic\\SenseVoiceSmall"


class SpeechRecognizer:
    def __init__(self, model_dir=SENSE_VOICE_PATH, device="cuda:0"):
        self.model = AutoModel(
            model=model_dir,
            # trust_remote_code=True,
            # remote_code="./model.py",
            # vad_model="fsmn-vad",
            # vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True
        )

    def generate(self, audio_path):
        # 生成转录结果
        result = self.model.generate(
            input=audio_path,
            cache={},
            language="auto",  # 自动检测语言
            use_itn=True,  # 使用反正则化（数字转文字等）
            batch_size_s=60,  # 每批处理60秒音频
            merge_vad=True,  # 合并语音活动检测片段
            merge_length_s=15,  # 合并15秒内的相邻片段
        )

        # 应用后处理
        text = rich_transcription_postprocess(result[0]["text"])
        emotion = result[0]["text"].replace(text, '')
        return text, emotion
