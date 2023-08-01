from human_eval.data import write_jsonl, read_problems
import torch, sys
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig #causalLM: 因果语言模型
from transformers import PreTrainedModel, PreTrainedTokenizer
from utils.prompter import Prompter
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel
)

adapter_url = r"adapter/python_humaneval/10"
base_model = r"/model_bak"

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=1,
    top_k=50,
    num_beams=4,
    max_new_tokens=200,
    stream_output=False,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield prompter.get_response(output)

device_map = "auto"
tokenizer = LlamaTokenizer.from_pretrained(base_model)

prompter = Prompter("alpaca")

model = LlamaForCausalLM.from_pretrained(
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    cache_dir="./cache_dir",
    pretrained_model_name_or_path=base_model
)
model = PeftModel.from_pretrained(
            model,
            adapter_url,
            torch_dtype=torch.float32,
        )

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

while(True):
    inp = input("input\n")
    for item in evaluate(inp):
        print(repr(item))
    print("\n")
