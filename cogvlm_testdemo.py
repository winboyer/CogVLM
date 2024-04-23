import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

print('tokenizer')
# tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
tokenizer = LlamaTokenizer.from_pretrained('/home/ubuntu/models/vicuna-7b-v1.5')
print('AutoModelForCausalLM')
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        '/home/ubuntu/models/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
# # 2 3090 默认分配
# device_map = infer_auto_device_map(model, max_memory={0:'20GiB',1:'20GiB','cpu':'16GiB'}, no_split_module_classes='CogVLMDecoderLayer')
# 2 A10 尝试分配
device_map = infer_auto_device_map(model, max_memory={0:'19GiB',1:'20GiB','cpu':'17GiB'}, no_split_module_classes='CogVLMDecoderLayer')
model = load_checkpoint_and_dispatch(
    model,
    '/home/ubuntu/models/cogvlm-chat-hf',   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
    device_map=device_map,
)
model = model.eval()

# check device for weights if u want to
for n, p in model.named_parameters():
    print(f"{n}: {p.device}")


print('chat example')
# chat example
query = 'Describe this image'
image = Image.open('./examples/1.png').convert('RGB')
# image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true', stream=True).raw).convert('RGB')
print('build_conversation_input_ids')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

print('model.generate')
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))

