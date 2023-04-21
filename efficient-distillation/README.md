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
| BERT Base |  [66.4](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/tqzux47h) |  [88.1](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/7i9jk5x7) | [55.0](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/5t6wzhx0) | [93.0](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/tqzux47h)  | [90.7](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/w5xy7qj5) |  90.2  |      |
| BERT Large| [72.6](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/ieidolnr)      | [89.1](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/6y3nieqs)     |  [61.3](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/abwmwqi2)    |  [94.2](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/0it7ycbr)     | [92.4](https://wandb.ai/dogtooooth/bert-large-glue-distillation/runs/tjkd4zj9/)    |     |      |

### IWSLT 14 Machine Translation X-En
|   | De | Ar | Fa | Es | He |
|---|:--:|:--:|:--:|:--:|:--:|
|transformer_iwslt_de_en | 32.75 | 28.95 | 38.74| 19.71| 34.30|

## Self-Teaching BERT-base
Embedding parameters = 768*(30000+2+512) = around 23M parameters 
Each layer about 7.5M parameters
| Model           | RTE  | QNLI   | CoLA | MRPC | SST-2| QQP  |Approx. # of Parameters  |
|-----------------|:----:|:------:|:----:|:----:|:----:|:----:|:------------------------:|
| Full 12 Layers  | 66.4 | 90.7   | 55.0 | 88.1 | 93.0 | 90.2 |         110M             |
| DistillBERT (Sanh et al. 2019)| 59.9| 89.2 | 51.3| 87.5| 91.3 | 88.5| 66M             | 
| First 8 Layers  | 65.7 | 89.1   | 55.2 | 86.2 | 91.9 | 90.2 | 58M                     |
| 7               | 63.5 | 88.1   | 51.2 | 85.6 | 91.3 | 89.9 | 51M                     |
|6                | 63.2 | 86.9   | 43.3 | 84.5 |     |     |44M                      | 
| 5               | 62.1 |       |     |     |     |     |36M                      | 
|  4              | 56.0 |       |     |     |     |    |29M                      | 


## Self-Teaching BERT-Large
Each layer around 13M parameters

| Model| QNLI | MRPC | RTE | SST-2 | Approx. # of Parameters |
|------|:----:|:----:|:----:|:----:|:----------------------:|
| Full 24 layers |  92.4 | 89.1 | 72.6 | 94.2 | 336M |
| First 12 layers|  87.8  | 84.6 | 53.4 | 89.1 | 183M |
| First 11 layers | - | - | 55.6 | 89.0 |  |
| First 10 layers |  85.3  | 81.1 | 54.5 | 88.2 | 158M |
| First 9 layers | - | - | 56.0 | 87.6 |  |
| First 8 layers | - | - | 56.0 | 87.6 |  |

## Gradual Self-Teaching BERT-Large 

Schedule: 24-20, 20-16, 16-12
| Model| CoLA (Mcc.) | RTE (Acc.) | Approx. # of Parameters |
| -----|:-----------:|:----------:|:-----------------------:|
| Full 24 Layers  | 61.3 | 72.6   |        336M             |
| First 20 Layers | 58.6 | 71.1   |        254M             |
| First 16 Layers | 60.4 |  70.0  |         203M            |
| First 12 Layers | **58.1** |  **63.9**  |         183M            |
| First 12 Layers w/o gradual| 31.0 | 53.4 |         183M            |
    
