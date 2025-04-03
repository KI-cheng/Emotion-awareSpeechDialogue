# 4. Implementation

This section describes the technical implementation of the emotional dialogue system, and introduces in detail the operational configuration, parameters, and technical details involved in the proposed method.

## 4.1 Fine-tuning of small parameters and large language model

We choose Qwen2.5-7B[10] as the base model, whose 7B parameter volume shows an excellent balance between semantic understanding and generation tasks. To improve the model's emotion classification ability, supervised fine-tuning is performed using the CARER[11] dataset, which contains 416,809 text samples with six emotion labels (sadness, joy, love, anger, fear, surprise). All the codes in this project are run in Google Colab, and fine-tuned using a Tesla T4 graphics card. The local deployment environment is an RTX 2060 laptop, Pytorch 2.6, CUDA 12.6, unsloth-window. The training process is monitored using W&B. The code implementation fully covers the entire process of data cleaning, model training, and evaluation and testing to ensure the reproducibility of the experiment.

### 4.1.1 Memory optimization configuration

In order to improve the inference speed, this project uses the unsloth framework [12] for large model fine-tuning. Unsloth is a framework specifically for fine-tuning LLMs. It optimizes the fine-tuning of large language models, making their training speed 2 times faster and reducing memory usage by 70%.
Introduce the model and lora adapter settings through FastLanguageModel. Use 4-bit quantization (load_in_4bit=True) to reduce video memory consumption and support training on 24GB video memory devices. Automatically select bfloat16/float16 precision according to the hardware architecture. Tesla T4 uses float16 and A100 enables bfloat16.

### 4.1.2 LoRA parameter settings

Use a low-rank adapter with rank 16 (r=16), scaling factor α=16, and the target module covers the attention mechanism (q_proj, k_proj, v_proj, o_proj) and MLP layer (gate/up/down_proj). Disable LoRA dropout (lora_dropout=0) to maintain optimal performance, and reduce video memory usage through gradient checkpoints (unsloth optimized version).

### 4.1.3 Training configuration

Set the maximum sequence length to 2048 to support long text processing, use AdamW 8bit optimizer, learning rate 2e-4 with linear decay strategy. Use gradient accumulation (gradient_accumulation_steps=4) and small batch training (per_device_batch_size=2), and complete 1 epoch training after 40 steps of warm-up.

### 4.1.4 Fine-tuning instruction construction

Build instruction templates to convert raw text into structured prompts:

```python

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Then there are five types: anger, fear, joy, love, sadness, and surprise. You need to determine which category the emotion of input belongs to and return.

### Input:
{}

### Response:
{}"""
```

## 4.2 Training Results and Performance Evaluation

### 4.2.1 Loss

![cross_entropy](cross_entropy_loss.png)
![training_loss](training_loss.png)
- Fig1 Training and test loss

Initial stage (0-10 steps): The loss value starts at about 2.6, reaches a peak of about 2.8 around the second step, and then drops rapidly. This shows that the model learns very quickly in the early stages.

Mid-stage (10-20 steps): The loss value begins to fluctuate between 0.6-1.0, but the overall trend is still decreasing.

Late stage (20-60 steps): The loss value oscillates between 0.6-1.1, but is basically stable, indicating that the model is close to convergence.

But the overall trend shows that the training is successful. As the training continues, the fluctuation of the loss value decreases, proving that the model gradually finds a stable parameter space.

### 4.2.2 GradNorm

![training_grad_norm](training_grad_norm.png)
- Fig2 Gradnorm values

At the beginning of training (0-10 steps), the grad_norm value fluctuates between 0.8-1.0 and is relatively stable.

At about the 10th step, the light blue line has a clear peak of about 1.5, which may indicate that a large gradient change occurred during this training phase.

After the peak, the grad_norm value drops rapidly and stabilizes at about 0.3-0.4 between 15-20 steps.

As training continues (20-60 steps), the grad_norm value continues to slowly drop and stabilizes at a low value of about 0.25-0.3

This trend shows that the gradient is effectively normalized during training, and the training process becomes more stable.

### 4.2.3 Multitasking test

In addition to the main target of the model, sentiment classification, the model's dialogue continuity and modal particle segmentation dialogue capabilities were also verified. The experimental results show that the model has successfully learned the ability to recognize sentiment. The test set evaluation adopts a strict and precise matching strategy. The generated result is judged to be correct when it is completely consistent with the label. The final accuracy rate is 87%, which is much better than the baseline performance of the basic model. And 0 errors were achieved in fixed-format replies. However, in the latter two tests, the model results were not ideal, the diversity of answers was limited, and some error recovery appeared in the test results. It is speculated that there are two main reasons: First, the model parameters are small, and it is impossible to maintain the original excellent dialogue ability while completing the parameter adjustment task. Second, the fixed-format replies adjusted by parameters affect the generation ability of the model, and catastrophic forgetting occurs. This is also one of the reasons why we decided to introduce GPT to achieve content understanding. As for the ability to use modal particles to segment dialogues, it may be necessary to train a new model to achieve this task, so as to better affect the generation effect of TTS.

## 4.3 Reasoning optimization and deployment
The fine-tuned model optimizes the effect of the model in actual application in the following ways. Integrate TextStreamer to achieve token-level streaming output, greatly reducing response latency. Provide multiple deployment solutions. For example, export in GGUF format, provide multi-level quantization files such as q4_k_m (4.5GB), q8_0 (7.8GB) to deploy through llama.cpp. You can also accelerate the model inference speed by setting up an unsloth environment, and deploy by loading the adapter loral_model (169M) during fine-tuning.

## 4.4 Module Integration and Environment Management and Communication

In view of the heterogeneous dependencies between modules, we use Anaconda to establish different virtual environments to ensure compatibility. Inter-module communication is implemented through the Client-Server architecture, relying on the request and fastAPI libraries, running the entire TTS module as a service, and exchanging requests and responses through HTTP. This modular design enhances scalability and maintainability, allowing each component to be updated independently without affecting the entire system. The ASR module uses FunASR and WebtrcVAD to achieve real-time monitoring and recording of user voice, converting the user's words into corresponding WAV files and text, and reducing noise interference to a certain extent by detecting the environmental sound threshold. The TTS module uses the voiceprint definition provided by Fish-Speech to customize the effect of synthesized audio. The WAV format audio file played by PyAudio ensures consistency with the dialogue sequence.
