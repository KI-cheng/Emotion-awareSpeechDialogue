# 2. Related Work

## 2.1 Multimodal Interaction System

Multimodal emotional dialogue systems usually support multiple input and output methods such as text, voice and facial expressions, and can process data of multiple modalities at the same time. This multimodal fusion capability enables the system to understand and express emotions more comprehensively, thereby improving the quality of dialogue [1]. The training of multimodal systems requires a large amount of multimodal data to capture the complex relationship between different modalities. However, obtaining and annotating this data is a time-consuming and expensive task, and training the model requires a lot of time and computing resources. [2] Therefore, this study hopes to further decompose the emotional dialogue system to achieve higher accuracy and effect in a low-cost way, making low-cost local deployment possible.

## 2.2 Large Model Fine-tuning Technology

Parameter Efficient Fine-tuning (PEFT) technology [3] significantly reduces the computational and storage costs during training by freezing most of the parameters of the pre-trained backbone network, and has become an important method for optimizing large language models in recent years. Low-Rank Adaptation(LoRA), as a representative technology of PEFT, achieves efficient parameter updates through low-rank matrix decomposition, and only introduces a small number of trainable parameters to the model weights, thereby significantly reducing resource requirements while maintaining model performance. At the same time, supervised fine-tuning (SFT) significantly enhances the model's alignment with specific tasks through an instruction-based fine-tuning strategy, enabling it to better adapt to the needs of downstream tasks. Based on this technology, this project designs an efficient fine-tuning solution for the emotional dialogue system.

## 2.3 Automatic Speech Recognition

Automatic Speech Recognition (ASR) technology is an important component of multimodal emotional dialogue systems. It is responsible for converting user voice input into text, providing a basis for subsequent sentiment analysis and dialogue generation. In recent years, deep learning and large-scale datasets have promoted significant progress in ASR technology. For example, the Conformer model based on the Transformer architecture [4] performs well in speech recognition tasks, improving accuracy and robustness. In addition, self-supervised learning methods such as wav2vec 2.0 [5] reduce annotation costs and promote technology development by pre-training on unlabeled data. FunASR[6] is a framework that aims to connect academic research with industrial applications. It provides models pre-trained on large-scale industrial corpora and supports tasks such as speech recognition, voice activity detection, and punctuation recovery.

## 2.4 Text-to-speech

Text-to-Speech (TTS) technology is responsible for converting text responses into natural and emotional speech output in emotional dialogue systems. In recent years, end-to-end models such as Tacotron2[7] have significantly improved the naturalness and fluency of speech, while the neural network-based vocoder WaveNet[8] has further enhanced speech fidelity by modeling audio waveforms. Fish-speech[9] has achieved amazing improvements in synthesis speed, making application-level deployment of TTS possible.
