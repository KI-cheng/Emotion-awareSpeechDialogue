import time
from unsloth import FastLanguageModel
from transformers import TextStreamer
import warnings

warnings.filterwarnings("ignore")


class unsloth_Qwen:
    def __init__(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="./lora/lora_model",
            max_seq_length=2048,
            dtype=None,
            device_map="cuda",
            load_in_4bit=True,
        )
        self.model.eval()
        FastLanguageModel.for_inference(self.model)
        print("Qwen模型加载完毕...")

    def identification(self, input_text="He actually stole my business. It really pissed me off！"):
        prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        Then there are five types: anger, fear, joy, love, sadness, and surprise. You need to determine which category the emotion of input belongs to and return.

        ### Input:
        {}

        ### Response:
        {}"""
        inputs = self.tokenizer(
            [
                prompt.format(
                    input_text,
                    "",
                )
            ], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        output = self.model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=False
        )
        new_tokens = output.sequences[0, inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return generated_text
