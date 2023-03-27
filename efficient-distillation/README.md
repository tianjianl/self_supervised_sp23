## Getting Started
`pip install -r requirements.txt`

To download all of the GLUE data: 

`python3 data_utils/download_glue_data.py`

## Training
You can find training scripts at `scripts` folder

## Baselines

| Model     | RTE   | MRPC | CoLA | SST-2 |QNLI  | QQP | Avg. |
|-----------|-------|-------|------|-------|------|-----| ---- |
| BERT Base |  [66.4](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/tqzux47h) |  [88.1](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/7i9jk5x7) | [55.0](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/5t6wzhx0) | [93.0](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/tqzux47h)  | [90.7](https://wandb.ai/dogtooooth/bert-glue-distillation/runs/w5xy7qj5) |     |      |
| BERT Large|       |       |      |       |      |     |      |


## Self-Teaching RTE
Embedding parameters = 768*(30000+2+512) = around 23M parameters 
| Model  | Acc. | Appox. # of Parameters  |
|------|:----:|:------------------------:|
| Full 12 Layers  | 66.4 |           110M          |
| First 8 Layers| 65.7  | 58M |
| 7| 63.5 | 51M |
|6 | 63.2| 44M | 
| 5| 62.1 | 36M | 
| 4| 56.0 | 29M |

    
