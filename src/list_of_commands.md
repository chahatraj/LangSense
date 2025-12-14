```bash
"python tools/preprocess_data.py \
 --tokenizer-name-or-path meta-llama/Llama-3.2-1B \
 --output-folder /scratch/craj/langsense/datasets/base/train \
 --n-tasks 8 \
 local \
 --dataset /scratch/craj/langsense/data/base/hf_dataset/train \
 --column text"
```


```bash
"python tools/preprocess_data.py \
 --tokenizer-name-or-path meta-llama/Llama-3.2-1B \
 --output-folder /scratch/craj/langsense/datasets/base/validation \
 --n-tasks 8 \
 local \
 --dataset /scratch/craj/langsense/data/base/hf_dataset/validation \
 --column text"
```




```bash
"python tools/preprocess_data.py \
 --tokenizer-name-or-path meta-llama/Llama-3.2-1B \
 --output-folder /scratch/craj/langsense/datasets/valence_lang/categorical/train \
 --n-tasks 8 \
 local \
 --dataset /scratch/craj/langsense/data/new_langs/valence_lang/categorical/hf_dataset/train \
 --column text"
```


```bash
"python tools/preprocess_data.py \
 --tokenizer-name-or-path meta-llama/Llama-3.2-1B \
 --output-folder /scratch/craj/langsense/datasets/valence_lang/categorical/validation \
 --n-tasks 8 \
 local \
 --dataset /scratch/craj/langsense/data/new_langs/valence_lang/categorical/hf_dataset/validation \
 --column text"
```



```bash
python slurm_launcher.py \
 --gpus_per_node 1 \
 --partition contrib-gpuq \
 --qos gpu \
 --time_limit 0-06:00:00 \
 --enable-wandb \
 --no-sanity \
 --run valence_cat \
 --dp 1 \
 --tp 1 \
 --pp 1 \
 --mbs 128 \
 --acc 2 \
 --model 160m \
 --vocab-size 128256 \
 --tokenizer meta-llama/Llama-3.2-1B \
 --save-interval 500 \
 --grad-clip 0.1 \
 --learning-rate 1e-3 \
 --min-lr 1e-4 \
 --weight-decay 0.033 \
 --warmup-steps 500 \
 --seq 128 \
 --steps 6000 \
 --dataset /scratch/craj/langsense/datasets/valence_lang/categorical/train \
 --validation-dataset /scratch/craj/langsense/datasets/valence_lang/categorical/validation \
 --val-check-interval 500 \
 --slurm-logs-path /scratch/craj/langsense/logs/slurm_logs \
 --checkpoints-path /scratch/craj/langsense/logs/checkpoints \
 --configs-path /scratch/craj/langsense/logs/configs \
 --slurm-scripts-dir /scratch/craj/langsense/logs/slurm_scripts \
 --auto-resume
```

```bash
python slurm_launcher.py \
 --gpus_per_node 1 \
 --partition contrib-gpuq \
 --qos gpu \
 --time_limit 0-06:00:00 \
 --enable-wandb \
 --no-sanity \
 --run base_lm \
 --dp 1 \
 --tp 1 \
 --pp 1 \
 --mbs 128 \
 --acc 2 \
 --model 160m \
 --vocab-size 128256 \
 --tokenizer meta-llama/Llama-3.2-1B \
 --save-interval 500 \
 --grad-clip 0.1 \
 --learning-rate 1e-3 \
 --min-lr 1e-4 \
 --weight-decay 0.033 \
 --warmup-steps 500 \
 --seq 128 \
 --steps 6000 \
 --dataset /scratch/craj/langsense/datasets/base/train \
 --validation-dataset /scratch/craj/langsense/datasets/base/validation \
 --val-check-interval 500 \
 --slurm-logs-path /scratch/craj/langsense/logs/slurm_logs \
 --checkpoints-path /scratch/craj/langsense/logs/checkpoints \
 --configs-path /scratch/craj/langsense/logs/configs \
 --slurm-scripts-dir /scratch/craj/langsense/logs/slurm_scripts \
 --auto-resume
```


```bash
~/nanotron-env/bin/python -m torch.distributed.run \
 --nproc_per_node=1 convert_to_hf.py \
--checkpoint_path=/scratch/craj/langsense/logs/checkpoints/valence_cat/3000\
 --save_path=/scratch/craj/langsense/models/valence_cat
```


```bash
python evaluate_perplexity.py --model-path /scratch/craj/langsense/models/valence_tag \
--dataset-path /scratch/craj/langsense/data/base/hf_dataset/test
```


```bash
python sft.py --modelname base_lm
```