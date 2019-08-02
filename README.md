Code for our ACL 2019 paper:

## Self-Supervised Learning for Contextualized Extractive Summarization

Paper link: [https://arxiv.org/abs/1906.04466](https://arxiv.org/abs/1906.04466)

### Prepare data

Download [data](https://drive.google.com/file/d/1oHrS23qQs0ufcmSQAMAlg-2wf1xBuyT6/view?usp=sharing)

### Requirement

Install [pyrouge](https://github.com/bheinzerling/pyrouge)

### pretrain the model

```
sh run_pretrain.sh
```

### fine-tune the model

```
sh run_summarization.sh
```

### evaluate the model

copy the model under 'results/models' to 'results/model_to_evaluate.pt'
```
sh run_eval.sh
```
