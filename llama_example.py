# transfromers version 4.32.0
import warnings
warnings.filterwarnings("ignore")

import llama_self_extend_patch as LlamaSE
import llama_self_extend_patch_hip as LlamaHipSE
from modify_utils import modify_method_of_instance
from functools import partial
import json
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

original_llama_forward = LlamaAttention.forward
self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=4, group_size_2=2048)
hip_forward = partial(LlamaHipSE.self_extend_forward, group_size_1=4, group_size_2=2048)

model_path = 'meta-llama/Llama-2-7b-chat-hf'
# model_path = 'princeton-nlp/Sheared-LLaMA-1.3B'
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map={'':'cuda:0'},
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

for idx, layer in enumerate(model.model.layers):
    layer.self_attn.layer_index = idx

eval_vanila = False
eval_self_extend = False
eval_hip = True

for line in open("passkey_examples_10k.jsonl", "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    # prompt = example["input"][5000:] + prompt_postfix
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(0)
    print( "-----------------------------------" )
    print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
    print( "Passkey target:", example["target"] )

    if eval_vanila:
        modify_method_of_instance(model, "LlamaAttention", "forward", original_llama_forward)
        tokens = model.generate(input_ids, max_new_tokens=6)
        answer= "Llama2:     [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
        answer = answer.replace("\n", "\\n")
        print( answer )

    if eval_self_extend:
        modify_method_of_instance(model, "LlamaAttention", "forward", self_extend_forward)
        tokens = model.generate(input_ids, max_new_tokens=6)
        answer= "SelfExtend: [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
        answer = answer.replace("\n", "\\n")
        print( answer )
    
    if eval_hip:
        modify_method_of_instance(model, "LlamaAttention", "forward", hip_forward)
        tokens = model.generate(input_ids, max_new_tokens=6)
        answer= "Hip:        [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
        answer = answer.replace("\n", "\\n")
        print( answer )
    print( "-----------------------------------\n" )