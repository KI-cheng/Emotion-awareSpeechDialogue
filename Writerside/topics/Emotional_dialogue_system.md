# Emotional dialogue system based on large language model

## Abstract

Emotional dialogue system has important application value in the field of human-computer interaction, but existing methods often face problems such as low multimodal collaboration efficiency and coarse granularity of emotion understanding. This study proposes a multimodal emotional dialogue system based on a large model, which realizes emotional state reasoning and empathetic response generation by fine-tuning the Qwen2.5-7B model, and combines dynamic voice interaction technology to build a complete dialogue logic. The system adopts LoRA parameter freezing and SFT fine-tuning strategies to achieve a loss value of 0.6 and a personality recognition accuracy of 87% on the emotion dataset; it integrates ASR module (supports multi-language real-time monitoring), GPT agent module (prompt word engineering drives empathetic response) and TTS module (voiceprint customized speech synthesis), and realizes module decoupling through client-server architecture. Experiments show that the system performs well in terms of emotional response naturalness and multimodal collaboration efficiency, providing a reproducible practical framework for lightweight emotional computing systems.

## Keywords

Emotional dialogue system, large model fine-tuning, LoRA, speech recognition (ASR), speech synthesis (TTS), prompt word engineering, multimodal interaction.
