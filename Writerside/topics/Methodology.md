# 3 Methodology

![structure](structure.jpg)

In this section, we describe the approach taken to build an affective dialogue system based on a large language model. The system senses and responds to user emotions through a multi-module architecture. **As shown in Figure 1**, the system consists of a real-time speech detection and automatic speech recognition (ASR) module, an emotion inference and empathetic response generation module, and a text-to-speech (TTS) module, forming a complete voice interaction loop.

## 3.1 Real-time Speech Detection and ASR Module

To facilitate voice-driven interactions, we integrated an ASR module to transcribe speech input into text. This module monitors and records user speech clips in real time and supports multiple languages, enhancing system accessibility. When speech clips are detected, they are dynamically converted into text transcriptions and fed into the subsequent sentiment analysis. This design ensures a seamless integration with the core emotion recognition framework.  
**In the upper red section of Figure 1**, you can see the detailed workflow of "Voice Detection," "Audio Transcription," and the "Lightweight ASR Model," where user speech inputs are detected in real time and transcribed using a lightweight ASR, enabling subsequent emotion inference and dialogue generation.

## 3.2 Emotion Inference and Empathetic Response Generation

The core functionality of the system lies in its ability to infer user emotions and generate empathetic responses. We developed an architecture with multiple large language models (LLMs) operating in concert to enable this capability. This part of the system is implemented by combining locally deployed, fine-tuned smaller-parameter LLMs with large-parameter LLM APIs for more nuanced understanding of emotions and text content. The use of prompt engineering further improves LLM understanding of user emotions and conversation context, resulting in highly empathetic responses.  
**In the lower orange section of Figure 1**, the key modules of emotion inference and empathetic response generation are shown, including components such as the "Emotion Classifier," "Prompt Engineering," and "Empathetic Response," which work together to deliver efficient emotion recognition and high-quality dialogue generation.

## 3.3 Text-to-Speech (TTS) Module

Since the overall system architecture starts with the user’s voice input, providing voice output can significantly enhance the user experience. In this module, we convert the LLM-generated responses into spoken content and return them to the user. Combined with audio playback, the system completes the loop of the conversational logic.  
**As indicated by the yellow module in Figure 1**, after generating the emotion label and corresponding response, the system sends the output text to a personalized TTS module configured with a target voice profile. This enables the final synthesized voice output, closing the affective dialogue loop.
