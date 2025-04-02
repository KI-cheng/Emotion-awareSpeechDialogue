import time

from llama_cpp import Llama


class Qwen:
    def __init__(self):
        model_path = "gguf/unsloth.Q4_K_M.gguf"
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048
        )

    def identification(self, input_text="Hi, what's going on?"):
        self.model.reset()
        prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
        ### Instruction:
        There are five types: anger, fear, joy, love, sadness, and surprise. You need to determine which category the emotion of input belongs to and return.
    
        ### Input:
        {}
    
        ### Response:
        {}""".format(input_text, "")

        output = self.model(
            prompt,
            max_tokens=512,
            temperature=0.7,
        )
        return output["choices"][0]["text"]

    def generation_response(self, input_text="Hi,my name is Ben.", input_emotion="happy"):
        # 构建提示
        prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                You must return a short reply to the user's input

                ### Input:
                {}

                ### Response:
                {}""".format(input_text, "")

        # 转换为token序列
        tokens = self.model.tokenize(prompt.encode())

        # 流式生成
        response_text = ""
        for token in self.model.generate(
                tokens,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1
        ):
            piece = self.model.detokenize([token]).decode()
            response_text += piece
            print(piece, end="", flush=True)
        return response_text


q = Qwen()
first = time.time()
print(first)
res = q.generation_response()
print(res)
print(time.time() - first)
