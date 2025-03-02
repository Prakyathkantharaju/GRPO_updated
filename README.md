





<!-- ```bash
docker run --gpus "device=0" -p 8080:80 -v $PWD/data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id BAAI/bge-large-en-v1.5 --auto-truncate --max-client-batch-size 128
``` -->


# Create datasets:


## Clone the repo:

```bash
git clone https://github.com/teknium1/GPTeacher.git
```


## Create datasets:

- Create a folder called dataset:
and run the following command:
```bash
python create_datasets.py
```



# Run the main script:

```bash
accelerate launch --num_processes 7 --config_file deepspeed_zero1.yaml scripts/grpo_training_2.py --config grpo-qwen-2.5-r1.yaml
```


