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
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from sat.quantization.kernels import quantize
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.generation.autoregressive_sampling import filling_sequence, stream_filling_sequence, get_masks_and_position_ids_default
from utils.vision import get_image_processor

# from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

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

world_size = int(os.environ.get('WORLD_SIZE', 1))

model_chat, model_chat_args = CogVLMModel.from_pretrained(
    '/home/ubuntu/.sat_models/cogvlm-chat-v1.1',
    args=argparse.Namespace(
    deepspeed=None,
    local_rank=0,
    rank=0,
    world_size=world_size,
    model_parallel_size=world_size,
    mode='inference',
    fp16=args.fp16,
    bf16=args.bf16,
    skip_init=True,
    use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
    device='cpu' if args.quant else 'cuda'),
    overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
)
model_chat = model_chat.eval()
assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

tokenizer = llama2_tokenizer('/home/ubuntu/models/vicuna-7b-v1.5', signal_type=args.version)
image_processor = get_image_processor(model_chat_args.eva_args["image_size"][0])

if args.quant:
    quantize(model_chat, args.quant)
    if torch.cuda.is_available():
        model_chat = model_chat.cuda(0)

model_chat.add_mixin('auto-regressive', CachedAutoregressiveMixin())
text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model_chat.image_length)

# model_grounding, model_grounding_args = CogVLMModel.from_pretrained(
#     '/home/ubuntu/models/CogVLM/cogvlm-grounding-generalist-v1.1',
#     args=argparse.Namespace(
#     deepspeed=None,
#     local_rank=0,
#     rank=0,
#     world_size=world_size,
#     model_parallel_size=world_size,
#     mode='inference',
#     fp16=args.fp16,
#     bf16=args.bf16,
#     skip_init=True,
#     use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
#     device='cpu' if args.quant else 'cuda'),
#     overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
# )
# model_grounding = model_grounding.eval()

# if args.quant:
#     quantize(model_grounding, args.quant)
#     if torch.cuda.is_available():
#         model_grounding = model_grounding.cuda(1)

# model_grounding.add_mixin('auto-regressive', CachedAutoregressiveMixin())

early_stop = False

def process_video(file_path, img_processor, interval_time=4.0):
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
        img_dict = img_processor(pil_img)
        ret = (img_dict, pil_img)
        ret_list.append(ret)

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
            invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else []
            repetition_penalty=1.0
            strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor_infer.tokenizer.eos_token_id],
                                    invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
            prompt = text_processor_infer.history_to_prompt(input_text, result_text)

            image_list = process_video(file_prompt.name, image_processor)
            ret_img_list = []
            response_list = []
            ret_cnt=0
            for torch_image, pil_img in image_list:
                if early_stop:
                    break
                if torch_image is not None:
                    for k in torch_image:
                        if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                            torch_image[k] = torch_image[k].to(torch.bfloat16 if args.bf16 else torch.float16)
                        if type(torch_image[k]) is torch.Tensor:
                            torch_image[k] = torch_image[k].to(next(model_chat.parameters()).device)
                # chat
                img_inputs = {'vision_'+k: v for k, v in torch_image.items()}
                
                inputs_dic = text_processor_infer(prompt)
                for k in inputs_dic:
                    if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
                        inputs_dic[k] = inputs_dic[k].to(torch.bfloat16 if args.bf16 else torch.float16)
                    if type(inputs_dic[k]) is torch.Tensor:
                        inputs_dic[k] = inputs_dic[k].to(next(model_chat.parameters()).device)
                input_ids = inputs_dic['input_ids'].to(model_chat.parameters().__next__().device)[0]
                if args.max_length-len(input_ids) <= 1:
                    response = "The prompt exceeds the context length limit, please try again."
                    return response, result_text, (torch_image, pil_img)

                seq = torch.cat(
                    [input_ids, torch.tensor([-1]*(args.max_length-len(input_ids)), device=input_ids.device)], dim=0
                )
                get_func = text_processor_infer.get_func(input_ids, **inputs_dic) if hasattr(text_processor_infer, 'get_func') else get_masks_and_position_ids_default
                
                inputs_dic.pop('input_ids')
                inputs = {**img_inputs, **inputs_dic}
                output = filling_sequence(
                    model_chat, seq,
                    batch_size=1,
                    get_masks_and_position_ids=get_func,
                    strategy=strategy,
                    **inputs
                )[0] # drop memory
                print('output==========', output)
                if type(output) is not list:
                    output_list = output.tolist()
                else:
                    output_list = output
                response = text_processor_infer.tokenizer.decode(output_list[0])
                print('response==========', response)
                if hasattr(text_processor_infer, 'process_response'):
                    response = text_processor_infer.process_response(response)
                    print('response==========', response)
                response = response.split(text_processor_infer.sep)[-1].strip()
                print('response==========', response)

                # grounding
                if ' no ' in response or 'not visible' in response:
                    # print('verb_noun_list, det_list========', verb_noun_list, det_list)
                    # history = history + [(query, response)]
                    continue
                ret_cnt += 1





                # inputs = model_grounding.build_conversation_input_ids(tokenizer, query=input_text, history=[], images=[pil_img])  # grounding mode
                # inputs = {
                #     'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                #     'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                #     'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                #     'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                # }
                # with torch.no_grad():
                #     outputs = model_grounding.generate(**inputs, **gen_kwargs)
                #     outputs = outputs[:, inputs['input_ids'].shape[1]:]
                #     # print(tokenizer.decode(outputs[0]))
                #     response_g = tokenizer.decode(outputs[0])
                #     response_g = re.sub('<pad>|<s>|</s>|<EOI>', '', response_g)
                # print('response_g=========', response_g)
                # bbox_num, draw_img = draw_response2img(pil_img, response_g)
                # if bbox_num<1:
                #     continue
                # gallery_prompt.append(draw_img)
                # answer = 'image '+str(ret_cnt)+': '+response
                # result_text.append((input_text, answer))
                # print(result_text)

                yield "", result_text, hidden_image, gallery_prompt

            # image_path_grounding = './results/output.png'
            # show_image(cache_image, image_path_grounding)
            # result_text.append((None, (image_path_grounding,)))

            
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
                result_text = gr.components.Chatbot(label='Single-round conversation History', 
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

