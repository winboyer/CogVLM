import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import requests
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import base64
import hashlib
import torch
import time
import re
import argparse
import cv2
import matplotlib.font_manager

from utils.parser import text_to_dict
from models.cogvlm_model import CogVLMModel
from sat.quantization.kernels import quantize
from utils.vision import get_image_processor

from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
parser.add_argument("--english", action='store_true', help='only output English')
parser.add_argument("--version", type=str, default="chat", help='version to interact with')
parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--server_port", type=int, default=7861, help='the gradio server port')
args = parser.parse_args()
parser = CogVLMModel.add_model_specific_args(parser)
args = parser.parse_args()   

if args.quant:
    tokenizer = LlamaTokenizer.from_pretrained('/home/ubuntu/models/vicuna-7b-v1.5')
    with init_empty_weights():
        model_chat = AutoModelForCausalLM.from_pretrained(
            '/home/ubuntu/models/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            load_in_4bit=args.quant == 4,
            load_in_8bit=args.quant == 8,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='balanced'   #'auto','balanced','balanced_low_0','sequential'
        )
        model_grounding = AutoModelForCausalLM.from_pretrained(
            '/home/ubuntu/models/cogvlm-grounding-generalist-hf',
            # '/home/ubuntu/models/cogvlm-grounding-generalist-v1.1',
            torch_dtype=torch.bfloat16,
            load_in_4bit=args.quant == 4,
            load_in_8bit=args.quant == 8,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='balanced'
        )
        # print('model_chat========', model_chat)
        # print('model_grounding========', model_grounding)
        # https://www.bilibili.com/read/cv25835999/
        # nf4_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        # model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

    # device_map = infer_auto_device_map(model_chat, max_memory={0:'16GiB', 1:'20GiB','cpu':'40GiB'}, no_split_module_classes='CogVLMDecoderLayer')
    # model_chat = load_checkpoint_and_dispatch(
    #     model_chat,
    #     '/home/ubuntu/models/cogvlm-chat-hf',
    #     device_map=device_map,
    # )
    model_chat = model_chat.eval()
    model_grounding = model_grounding.eval()
        
else:
    tokenizer = LlamaTokenizer.from_pretrained('/home/ubuntu/models/vicuna-7b-v1.5')
    with init_empty_weights():
        model_chat = AutoModelForCausalLM.from_pretrained(
            '/home/ubuntu/models/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model_grounding = AutoModelForCausalLM.from_pretrained(
            '/home/ubuntu/models/cogvlm-grounding-generalist-hf',
            # '/home/ubuntu/models/cogvlm-grounding-generalist-v1.1',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    # 2 A10 尝试分配
    # device_map = infer_auto_device_map(model_chat, max_memory={0:'16GiB',1:'20GiB','cpu':'18GiB'}, no_split_module_classes='CogVLMDecoderLayer')
    # device_map_g = infer_auto_device_map(model_grounding, max_memory={2:'16GiB',3:'20GiB','cpu':'18GiB'}, no_split_module_classes='CogVLMDecoderLayer')
    device_map = infer_auto_device_map(model_chat, max_memory={0:'16GiB',1:'20GiB','cpu':'18GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
    device_map_g = infer_auto_device_map(model_grounding, max_memory={2:'16GiB',3:'20GiB','cpu':'18GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])

    model_chat = load_checkpoint_and_dispatch(
        model_chat,
        '/home/ubuntu/models/cogvlm-chat-hf',
        device_map=device_map,
    )
    model_grounding = load_checkpoint_and_dispatch(
        model_grounding,
        '/home/ubuntu/models/cogvlm-grounding-generalist-hf',
        # '/home/ubuntu/models/cogvlm-grounding-generalist-v1.1',
        device_map=device_map_g,
    )

    model_chat = model_chat.eval()
    model_grounding = model_grounding.eval()

# # check device for weights if u want to
# for n, p in model.named_parameters():
#     print(f"{n}: {p.device}")

early_stop = False

def history_to_prompt(query, history):
    signal_type="chat"
    if signal_type == 'base':
        return '<EOI>' + query
    if signal_type == 'vqa':
        answer_format = 'Short answer:'
    else:
        answer_format = 'Answer:'

    prompt = '<EOI>'
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt

def process_video(file_path, interval_time=4.0):
    videoCapture = cv2.VideoCapture(file_path)
    success, frame = videoCapture.read()
    frame_cnt = 0
    # 视频中，1s包含25帧
    ret_list = []
    while success :
        frame_cnt = frame_cnt + 1
        if frame_cnt%(25*interval_time) != 0: # 隔4s抽一帧
            success, frame = videoCapture.read()
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # pil_img = pil_img.convert('RGB')
        ret_list.append(pil_img)
        success, frame = videoCapture.read()
    print('frame_cnt, len(ret_list)=========', frame_cnt, len(ret_list))
    videoCapture.release()

    return ret_list

def draw_response2img(img, response):
    img = img.convert('RGB')
    width, height = img.size
    ratio = min(1920 / width, 1080 / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    new_img = img.resize((new_width, new_height), Image.LANCZOS)

    response_dic = text_to_dict(response)
    if not response_dic:
        texts = []
        boxes = []
    else:
        texts, boxes = zip(*response_dic.items())
        
    width, height = new_img.size
    absolute_boxes = [[(int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)) for box in b] for b in boxes]
    
    color_palette = sns.color_palette("husl", len(absolute_boxes))
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in color_palette]

    overlay = Image.new('RGBA', new_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    # draw = ImageDraw.Draw(new_img)
    # print('len(boxes)===========', len(boxes))
    bbox_num = len(absolute_boxes)
    for box, color in zip(absolute_boxes, colors):
        # print(len(box[0]), box[0])
        # draw.rectangle([box[0][0], box[0][1], box[0][2], box[0][3]], outline=color, width=box_width)
        draw.rectangle(box[0], outline=color, width=5)
    
    img_with_overlay = Image.alpha_composite(new_img.convert('RGBA'), overlay).convert('RGB')

    return bbox_num, img_with_overlay


def web_pic_chat(file_path,
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 4096, top_p=0.95, top_k=5, temperature=0.95, repetition_penalty=1.0,
        invalid_slices=[], no_prompt=False):
    if image is None:
        assert file_path is not None
        image = Image.open(file_path).convert('RGB')
    else:
        filepath_extname = os.path.splitext(file_path)[-1]
        # print('filepath_extname, file_prompt============', filepath_extname, file_path)
        video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
        if not video_flag:
            image = Image.open(file_path).convert('RGB')

    if not history:
        history = []
    if no_prompt:
        query = ''
        
    ret_img_list = []
    response_list = []

    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}
    # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
    #                   "temperature": temperature, **kwargs}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        # print(tokenizer.decode(outputs[0]))

        response = tokenizer.decode(outputs[0])
        response = re.sub('<pad>|<s>|</s>|<EOI>', '', response)
        # print('response============', response)
    history = history + [(query, response)]

    response_list.append(response)
    ret_img_list.append(image)

    return response_list, history, ret_img_list


def web_video_chat(
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 4096, top_p=0.95, top_k=5, temperature=0.95, repetition_penalty=1.0,
        invalid_slices=[], no_prompt=False):
    
    if not history:
        history = []

    if no_prompt:
        query = ''
        
    print('total '+str(ret_cnt)+' images return!!!')
    return response_list, ret_img_list


default_chatbox = [("", "Hi, What do you want to know about?")]
def http_post(
        input_text,
        is_grounding,
        temperature,
        top_p,
        top_k,
        file_prompt,
        result_previous,
        hidden_image,
        ):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    gallery_prompt = []
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][0] == None:
            del result_text[i]
    print(f"history {result_text}")

    filepath_extname = os.path.splitext(file_prompt.name)[-1]
    # print('filepath_extname, file_prompt.name============', filepath_extname, file_prompt.name)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    global early_stop
    early_stop = False
    try:
        if video_flag:
            print('web_video_chat')
            image_list = process_video(file_prompt.name)

            ret_img_list = []
            response_list = []
            ret_cnt=0
            for pil_img in image_list:
                if early_stop:
                    break

                inputs = model_chat.build_conversation_input_ids(tokenizer, query=input_text, history=[], images=[pil_img])  # chat mode
                inputs = {
                    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                }
                gen_kwargs = {"max_length": 2048, "do_sample": False}
                # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                #                   "temperature": temperature, **kwargs}
                with torch.no_grad():
                    outputs = model_chat.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    # print(tokenizer.decode(outputs[0]))
                    response = tokenizer.decode(outputs[0])
                    response = re.sub('<pad>|<s>|</s>|<EOI>', '', response)
                    print('response============', response)

                # verb_noun_list, det_list = parse_query_response(response)
                # 判断 response中的关键字
                if ' no ' in response or 'not visible' in response:
                    # print('verb_noun_list, det_list========', verb_noun_list, det_list)
                    # history = history + [(query, response)]
                    continue
                ret_cnt += 1
                # print('response============', response)
                # response_list.append('image '+str(ret_cnt)+': '+response)

                inputs = model_grounding.build_conversation_input_ids(tokenizer, query=input_text, history=[], images=[pil_img])  # grounding mode
                inputs = {
                    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                }
                with torch.no_grad():
                    outputs = model_grounding.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    # print(tokenizer.decode(outputs[0]))
                    response_g = tokenizer.decode(outputs[0])
                    response_g = re.sub('<pad>|<s>|</s>|<EOI>', '', response_g)
                print('response_g=========', response_g)
                bbox_num, draw_img = draw_response2img(pil_img, response_g)
                if bbox_num<1:
                    continue
                gallery_prompt.append(draw_img)
                answer = 'image '+str(ret_cnt)+': '+response
                result_text.append((input_text, answer))
                print(result_text)

                yield "", result_text, hidden_image, gallery_prompt

            # image_path_grounding = './results/output.png'
            # show_image(cache_image, image_path_grounding)
            # result_text.append((None, (image_path_grounding,)))

        else:
            print('web_pic_chat no implement yet')

            response_list, _, image_list = web_pic_chat(
                file_path = file_prompt.name,
                query=input_text,
                history=result_text, 
                image=cache_image,
                max_length=2048, 
                top_p=top_p, 
                temperature=temperature,
                top_k=top_k,
                no_prompt=False
            )
            gallery_prompt.append(image_list[0])
            
    except Exception as e:
        print("error message", e)
        result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        # return "", result_text, hidden_image, gallery_prompt
        return "", result_text, hidden_image, gallery_prompt

    return "", result_text, hidden_image, gallery_prompt


def show_image(img, output_fn='./results/output.png'):
    img = img.convert('RGB')
    width, height = img.size
    ratio = min(1920 / width, 1080 / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    new_img = img.resize((new_width, new_height), Image.LANCZOS)

    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    img_with_overlay.save(output_fn)


def clear_fn(value):
    global early_stop
    early_stop = True
    return "", default_chatbox, None, None

def clear_fn2(value):
    return default_chatbox, None


def imagefile_upload_fn(file_prompt):
    filepath_extname = os.path.splitext(file_prompt.name)[-1]
    # print('filepath_extname, file_prompt.name============', filepath_extname, file_prompt.name)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    if not video_flag:
        return default_chatbox, file_prompt.name
    else:
        return default_chatbox, None

def change_gallery(result_text, gallery_prompt):

    return result_text, gallery_prompt

if __name__ == '__main__':
    gr.close_all()
    examples = []
    # is_grounding = False
    # example_ids = list(range(3)) if not is_grounding else list(range(3,6,1))
    # with open("./examples/example_inputs.jsonl") as f:
    #     for i, line in enumerate(f):
    #         if i not in example_ids: continue
    #         data = json.loads(line)
    #         examples.append(data)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=7):
                result_text = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
                hidden_image_hash = gr.Textbox(visible=False)

            with gr.Column(scale=4):
                # image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                # video_prompt = gr.Video(type="file", label="Video Prompt", value=None)
                file_prompt = gr.File(label="file prompt", 
                # file_types=["image",".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"], 
                file_types=[".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"], 
                value=None)
                gallery_prompt = gr.Gallery(label='chat image', height=300)

        with gr.Group():
            input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
            is_grounding = gr.Checkbox(label="Grounding")
            with gr.Row():
                run_button = gr.Button('Generate')
                clear_button = gr.Button('Clear')

            with gr.Row():
                temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                top_k = gr.Slider(maximum=50, value=5, minimum=1, step=1, label='Top K')
            
        # gr_examples = gr.Examples(examples=[[example["text"], example["image"]] for example in examples], 
        #                         inputs=[input_text, file_prompt],
        #                         label="Example Inputs (Click to insert an example into the input box)",
        #                         examples_per_page=6)
        
        print(gr.__version__)
        run_button.click(fn=http_post,inputs=[input_text, is_grounding, temperature, top_p, top_k, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        input_text.submit(fn=http_post,inputs=[input_text, is_grounding, temperature, top_p, top_k, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, file_prompt, gallery_prompt])
        file_prompt.upload(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # file_prompt.upload(fn=imagefile_upload_fn, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # file_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text, gallery_prompt])
        file_prompt.clear(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])

        # gallery_prompt.change(fn=change_gallery, inputs=[result_text, gallery_prompt], outputs=[result_text, gallery_prompt])

        print(gr.__version__)

    demo.queue(concurrency_count=10)
    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", show_error=True, server_port=args.server_port)
    # demo.launch(server_name="0.0.0.0", server_port=8088-8089)

