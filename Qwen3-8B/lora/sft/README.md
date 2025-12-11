---
library_name: peft
license: other
base_model: /data/lhx/LLaMA-Factory/models/base/Qwen3-8B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/data/lhx/LLaMA-Factory/models/base/Qwen3-8B](https://huggingface.co//data/lhx/LLaMA-Factory/models/base/Qwen3-8B) on the merged_extract_label_cot dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5104

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 2
- total_train_batch_size: 2
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.6507        | 0.1835 | 500  | 0.6555          |
| 0.5889        | 0.3669 | 1000 | 0.5883          |
| 0.5739        | 0.5504 | 1500 | 0.5656          |
| 0.5766        | 0.7338 | 2000 | 0.5530          |
| 0.525         | 0.9173 | 2500 | 0.5432          |
| 0.5286        | 1.1005 | 3000 | 0.5367          |
| 0.5439        | 1.2840 | 3500 | 0.5326          |
| 0.5437        | 1.4674 | 4000 | 0.5275          |
| 0.5092        | 1.6509 | 4500 | 0.5229          |
| 0.5094        | 1.8343 | 5000 | 0.5189          |
| 0.4738        | 2.0176 | 5500 | 0.5170          |
| 0.4744        | 2.2011 | 6000 | 0.5146          |
| 0.4818        | 2.3845 | 6500 | 0.5127          |
| 0.4882        | 2.5680 | 7000 | 0.5116          |
| 0.4843        | 2.7514 | 7500 | 0.5108          |
| 0.4853        | 2.9349 | 8000 | 0.5104          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.55.0
- Pytorch 2.8.0+cu128
- Datasets 3.6.0
- Tokenizers 0.21.4