from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lora/lora_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda"
)
FastLanguageModel.for_inference(model)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Then there are five types: anger, fear, joy, love, sadness, and surprise. You need to determine which category the emotion of input belongs to and return.

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
    [
        prompt.format(
            "He actually stole my business. It really pissed me off！！！！",
            "",  # output
        )
    ], return_tensors="pt").to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
