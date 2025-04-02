import pyaudio
import wave
import time
import webrtcvad
import os
import numpy as np

# 参数设置
AUDIO_RATE = 16000  # 音频采样率
AUDIO_CHANNELS = 1  # 单声道
CHUNK = 1024  # 音频块大小
VAD_MODE = 3  # VAD 模式 (0-3, 数字越大越敏感)
OUTPUT_DIR = "./reference"  # 输出目录
NO_SPEECH_THRESHOLD = 2  # 无效语音阈值，单位：秒

# 新增一个环境声音阈值
ENERGY_THRESHOLD = 300  # 能量阈值，低于此值的音频被视为噪音

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局变量
last_active_time = time.time()
segments_to_save = []
saved_intervals = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间
audio_file_count = 0

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)


# 新增：简单的音频能量计算
def calculate_energy(audio_data):
    """计算音频数据的能量"""
    data = np.frombuffer(audio_data, dtype=np.int16)
    return np.sqrt(np.mean(np.square(data.astype(np.float32))))


# 检测 VAD 活动 - 增加环境声音检测
def check_vad_activity(audio_data):
    # 先检查能量是否足够高
    energy = calculate_energy(audio_data)
    if energy < ENERGY_THRESHOLD:
        return False

    # 原有的 VAD 检测逻辑保持不变
    num, rate = 0, 0.4
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False


# 保存音频并返回保存的文件路径
def save_audio():
    global segments_to_save, last_vad_end_time, saved_intervals, audio_file_count

    if not segments_to_save:
        return None

    # 获取有效段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]

    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return None

    # 保存音频
    audio_file_count += 1
    audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"

    audio_frames = [seg[0] for seg in segments_to_save]

    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 16位PCM
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    # 记录保存的区间
    saved_intervals.append((start_time, end_time))

    # 清空缓冲区
    segments_to_save.clear()

    return audio_output_path


# 处理语音结果的函数（示例）
def process_audio_result(audio_file_path):
    print(f"正在处理音频文件: {audio_file_path}")
    # 这里添加您的语音处理逻辑
    # 例如：asr_result = your_asr_model.recognize(audio_file_path)

    # 模拟处理时间
    time.sleep(1)
    print("音频处理完成")


# 主函数 - 同步方式录制和处理
def record_and_process():
    global segments_to_save, last_active_time

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    audio_buffer = []
    print("音频录制已开始（按 Ctrl+C 停止）")

    try:
        while True:
            data = stream.read(CHUNK)
            audio_buffer.append(data)

            # 每 0.5 秒检测一次 VAD
            if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
                # 拼接音频数据并检测 VAD
                raw_audio = b''.join(audio_buffer)
                vad_result = check_vad_activity(raw_audio)

                if vad_result:
                    print("检测到语音活动")
                    last_active_time = time.time()
                    segments_to_save.append((raw_audio, time.time()))
                else:
                    print("无语音活动...")

                audio_buffer = []  # 清空缓冲区

            # 检查无效语音时间
            if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
                # 检查是否需要保存
                if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                    # 保存音频
                    audio_file_path = save_audio()

                    # 如果成功保存了音频文件，则处理它
                    if audio_file_path:
                        print(f"录制已停止，保存到{audio_file_path}")
                        return audio_file_path

                    last_active_time = time.time()

    except KeyboardInterrupt:
        print("录制停止中...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("录制已停止")