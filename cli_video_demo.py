# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin

from utils.video_chat import chat, video_chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat-v1.1", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    if not args.english:
        if rank == 0:
            print('欢迎使用 CogVLM-CLI ，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序')
    else:
        if rank == 0:
            print('Welcome to CogVLM-CLI. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')

    with torch.no_grad():
        while True:
            history = None
            cache_image = None
            if not args.english:
                if rank == 0:
                    image_path = [input("请输入图像路径或URL（回车进入纯文本对话）： ")]
                else:
                    image_path = [None]
            else:
                if rank == 0:
                    image_path = [input("Please enter the image path or URL (press Enter for plain text conversation): ")]
                else:
                    image_path = [None]
            if world_size > 1:
                torch.distributed.broadcast_object_list(image_path, 0)
            image_path = image_path[0]
            assert image_path is not None

            if image_path == 'stop':
                break
            if args.no_prompt and len(image_path) > 0:
                query = ""
            else:
                if not args.english:
                    if rank == 0:
                        query = [input("用户：")]
                    else:
                        query = [None]
                else:
                    if rank == 0:
                        query = [input("User: ")]
                    else:
                        query = [None]
                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
                query = query[0]
                assert query is not None
            
            image_path_extname = os.path.splitext(image_path)[-1]
            video_flag = (not image_path.startswith("http")) and image_path_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
            while True:
                if query == "clear":
                    break
                if query == "stop":
                    sys.exit(0)
                try:
                    if video_flag and cache_image is None:
                        response, history, cache_image = video_chat(
                            image_path,
                            model,
                            text_processor_infer,
                            image_processor,
                            query,
                            history=history,
                            image=cache_image,
                            max_length=args.max_length,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            invalid_slices=text_processor_infer.invalid_slices,
                            no_prompt=args.no_prompt
                            )
                    else:
                        response, history, cache_image = chat(
                            image_path,
                            model,
                            text_processor_infer,
                            image_processor,
                            query,
                            history=history,
                            image=cache_image,
                            max_length=args.max_length,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            invalid_slices=text_processor_infer.invalid_slices,
                            no_prompt=args.no_prompt
                            )
                except Exception as e:
                    print('Exception:', e)
                    break
                if rank == 0:
                    if not args.english:
                        print("模型："+response)
                        if tokenizer.signal_type == "grounding":
                            print("Grounding 结果已保存至 ./output.png")
                    else:
                        print("Model: "+response)
                        if tokenizer.signal_type == "grounding":
                            print("Grounding result is saved at ./output.png")
                image_path = None
                if not args.english:
                    if rank == 0:
                        query = [input("用户：")]
                    else:
                        query = [None]
                else:
                    if rank == 0:
                        query = [input("User: ")]
                    else:
                        query = [None]
                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
                query = query[0]
                assert query is not None


if __name__ == "__main__":
    main()
