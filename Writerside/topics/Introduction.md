# 1. Introduction

With the breakthrough of large language model technology, emotional intelligent dialogue system has gradually become a hot topic in human-computer interaction research. Traditional methods are mostly based on machine learning or deep learning to further improve the extraction of text information, but there are defects such as rigid response mode and insufficient context understanding. Although various large language models (LLMs) have demonstrated powerful generation capabilities, there are still limitations in the accuracy of emotional state recognition and personalized feedback due to the possibility of local deployment of large parameter models. There is still a lot of research space on how to balance the accuracy of emotion recognition, the empathy quality of generated responses, and computational efficiency.

This project proposes a lightweight multimodal emotional dialogue framework, and the core innovations include:

1. Adopting the efficient fine-tuning strategy of LoRA parameters, low loss value (0.6) and high personality recognition accuracy (87%) are achieved on the Qwen2.5-7B model;
2. Constructing a dynamic voice interaction pipeline, integrating ASR multi-language recognition and TTS voiceprint customization functions;
3. Designing a layered prompt word engineering, generating contextualized empathetic responses through the GPT proxy module.

The system reduces deployment complexity through modular design and environmental isolation, providing a reference for educational practice and industrial application.

