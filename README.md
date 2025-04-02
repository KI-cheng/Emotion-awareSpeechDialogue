# Emotion-Aware Speech Dialogue System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a lightweight, multimodal emotional dialogue system that leverages large language models to recognize emotions from speech, generate empathetic responses, and provide voice-based interactions. It is designed to run efficiently on consumer hardware, making advanced emotional AI accessible to a wider audience.

| **Aspects**           | **Traditional systems**                  | **Our system**                     |
| --------------------- | ---------------------------------------- | ---------------------------------- |
| Resource requirements | High  (High performance server required) | Low   (Supports consumer hardware) |
| Deployment            | Cloud services                           | Support local deployment           |
| Training efficiency   | Resource intensive, slow to train        | memory usage is reduced by 70%     |
| Deployment threshold  | High                                     | Low                                |
| Expandability         | Low (End-to-end)                         | High (Multiple modules)            |

## Features

- **Emotion Recognition:** Accurately identifies emotions from speech input using a fine-tuned Qwen2.5-7B model.(Accuracy:87%)
- **Empathetic Response Generation:** Generates contextually appropriate and empathetic responses based on recognized emotions.
- **Voice Interaction:** Supports real-time speech recognition and synthesis for natural conversations.
- **Multilingual Support:** Handles multiple languages for broader accessibility.
- **Efficient Fine-Tuning:** Utilizes LoRA for parameter-efficient fine-tuning, reducing resource requirements.
- **Modular Architecture:** Easy to extend and maintain with a client-server setup.
![training_and_evaluate_loss.png](fine_turning%2Ftraining_and_evaluate_loss.png)

## Technologies Used

- **Large Language Model:** [Qwen2.5-7B](https://github.com/QwenLM/Qwen2.5)
- **Fine-Tuning:**  [Unsloth🦥](https://github.com/unslothai/unsloth)), SFT, LoRA 
- **Speech Recognition:** [FunASR](https://github.com/modelscope/FunASR)
- **Speech Synthesis:** [Fish-Speech](https://github.com/fishaudio/fish-speech)
- **Framework:** PyTorch
- **Audio Processing:** webrtcvad,pyaudio
- **Module Communication:** FastAPI, Requests
- **Monitoring:** Weights & Biases (wandb)
![W&B Chart 2025_3_27 04_41_33.png](fine_turning%2FW%26B%20Chart%202025_3_27%2004_41_33.png)
![W&B Chart 2025_3_30 03_32_20.png](fine_turning%2FW%26B%20Chart%202025_3_30%2003_32_20.png)

## Setup and Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- **Python 3.10, 3.11, 3.12**: Ensure you have a compatible Python version installed. Because these are the versions which unsloth support.
- **Anaconda**: Used for managing virtual environments. Download it from [here](https://www.anaconda.com/products/distribution).
- **FFmpeg**: This is not necessary, but you can get faster speed by downloading it. After downloading, you need to configure it to the path in your system environment variables.
- **CUDA**:Our system use the 12.6. You can also use other versions without any conflicts. Please visit the pytorch official website to check the supported versions.
- **Pytorch**:Our system use the 2.60 version. Please select the corresponding version according to your cuda
- **Unsolth**:If you are using Windows, please follow our instructions to configure the environment. For other systems, please go to the guidance document on the unsloth official website for configuration.

### Clone the Repository

Clone the project repository to your local machine:

```
git clone https://github.com/KI-cheng/Emotion-awareSpeechDialogue.git 
cd Emotion-awareSpeechDialogue
```
**1.Install NVIDIA Video Driver**

You should install the latest version of your GPUs driver. Download drivers here:[NVIDIA GPU Drive](https://www.nvidia.com/en-us/drivers/)

**2.Install Visual Studio C++**

Launch the Installer here:  [Visual Studio Community Edition](https://visualstudio.microsoft.com/zh-hans/vs/community/)

**3.Install Python and CUDA Toolkit**

Follow the instructions to install[ CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).


#### Main Application Environment

Mainly used to load the unsloth environment. Due to the complexity of using unsloth on Windows, try to deploy it according to our version to avoid version conflicts.(trintion-windows need be awared that the Windows fork requires PyTorch >= 2.4 and CUDA 12)

```
conda create --name unsloth_env python=3.11
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
conda activate unsloth_env

pip install xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

#### ASR Module Environment

The environment layout is the same as the main module. You can download the following libraries directly in the above environment for convenience.
```
conda activate unsloth_env
pip install webrtcvad
pip install funasr
```

#### TTS Module Environment
Here we use the open source SOTA tts project fish-speech, which has efficient speech synthesis capabilities, extremely fast synthesis speed, and support for specific speech synthesis functions. Due to its modular design, you can also replace it with the TTS module you want to use.Here we introduce the environment construction of fish-speech.

```
git clone https://github.com/fishaudio/fish-speech
cd fish-speech
# You should download the modelweight from hf.
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5

conda create -n fish-speech python=3.10
conda activate fish-speech

pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip3 install -e .
```


## Usage
**Hardware Requirements:**

- **Training:** Colab T4. You can just open the fine-turing ipynb file to learn how it works.
- **Inference:** NVIDIA RTX 2060 6GB (Due to the author's hardware limitations, it can only run barely. Welcome everyone to run it on a better consumer-grade graphics card and tell me the feedback of the experience)

### Start the TTS Service

The TTS module runs as a separate service. Launch it first:

```
# start the original server and you can use our client to access.
python -m tools.api_server --listen 0.0.0.0:8080 --llama-checkpoint-path "checkpoints/fish-speech-1.5" --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" --decoder-config-name firefly_gan_vq --compile --half
```
**Note:** You may want to use --compile to fuse cuda kernels for faster inference (~30 tokens/sec -> ~500 tokens/sec).
Correspondingly, if you do not plan to use acceleration, you can comment out the --compile parameter.
For GPUs that don't support bf16, you may need to use the --half option.
(--compile still need c++ environments and cuda support.)

Finally, you can add the wav files as the target audio to the reference folder,  and the lab file describe the context of the wav file is need. All of them should be named as the same name.

### Run the Main Application

Start the main application, which handles ASR, emotion recognition, and response generation:

```
conda activate unsloth_env python main.py
```

### Interact with the System

- Once the model is loaded, speak into the microphone.
- The system will transcribe your speech, infer your emotion, generate an empathetic response, and play it back as audio.

#### Example Interaction

- **User Input:** "I'm feeling really stressed about my exams."
- **System Output:** [Recognize:sadness] (SPEECH)"I understand that exams can be overwhelming. Have you tried taking short breaks to relax?"


## License

This project is licensed under the MIT License - see the  file for details.

## Acknowledgments

- Thanks to the creators of Qwen2.5, LoRA, FunASR, Fish-Speech, and other open-source projects that enabled this work.
- Special gratitude to the contributors and the open-source community.