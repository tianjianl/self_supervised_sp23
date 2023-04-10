## Getting Started

To install the required dependencies:

`pip install -r requirements.txt`

To download all of the GLUE data: 

`python3 data_utils/download_glue_data.py`

## Training
You can find training scripts at `scripts` folder

## Baselines

### GLUE

| Model     | RTE   | MRPC | CoLA | SST-2 |QNLI  | QQP | Avg. |
|-----------|-------|-------|------|-------|------|-----| ---- |
| BERT Base |  [66.4](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/tqzux47h) |  [88.1](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/7i9jk5x7) | [55.0](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/5t6wzhx0) | [93.0](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/tqzux47h)  | [90.7](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/w5xy7qj5) |     |      |
| BERT Large| [72.6](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/ieidolnr)      | [89.1](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/6y3nieqs)     |  [61.3](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/abwmwqi2)    |  [94.2](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/0it7ycbr)     | [92.4](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/tjkd4zj9/)    |     |      |

### IWSLT 14 Machine Translation X-En
|   | De | Ar | Fa | Es | He |
|---|:--:|:--:|:--:|:--:|:--:|
|transformer_iwslt_de_en | 32.75 | 28.95 | 38.74| 19.71| 34.30|

## Self-Teaching BERT-base
Embedding parameters = 768*(30000+2+512) = around 23M parameters 
| Model  | Acc. | Approx. # of Parameters  |
|------|:----:|:------------------------:|
| Full 12 Layers  | 66.4 |           110M          |
| First 8 Layers| 65.7  | 58M |
| 7| 63.5 | 51M |
|6 | 63.2| 44M | 
| 5| 62.1 | 36M | 
| 4| 56.0 | 29M |


## Self-Teaching BERT-Large

| Model| QNLI | MRPC | RTE | Approx. # of Parameters |
|------|:----:|:----:|:|:----:|----------------------:|
| Full 24 layers |  92.4 | 89.1 | - |336M |
| First 12 layers|  87.8  | 84.6 | 53.4 | 183M |
| First 11 layers | - | - | 55.6 |  |
| First 10 layers |  85.3  | 81.1 | 54.5 | 158M |
| First 9 layers | - | - | 56.0 |  |
| First 8 layers | - | - | 56.0 |  |



    
