#CUDA_VISIBLE_DEVICES=2,3 python -u web_demo_grounding.py --version chat --english --bf16 --server_port 7861 > web_demo_grounding.log
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u web_demo_grounding.py --version chat --english --bf16 --server_port 7861 > web_demo_grounding.log
CUDA_VISIBLE_DEVICES=0,1 python basic_demo/cogagent_webdemo_quant.py --version chat --bf16 > cogagent_webdemo_quant.log

#CUDA_VISIBLE_DEVICES=2,3 python -u web_demo_grounding.py --version chat --english --bf16 --quant 8 --server_port 7861 > web_demo_grounding.log
#CUDA_VISIBLE_DEVICES=0,1 python -u web_demo_grounding_quant.py --version chat --english --fp16 --quant 8 --server_port 7862


