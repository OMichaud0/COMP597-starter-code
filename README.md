# COMP597-starter-code
This repository contains starter code for COMP597: Sustainability in Systems Design - Energy Efficiency analysis using CodeCarbon. 

### Instructions

1. Set up your environment using the provided instructions below under [Environment Setup](#environment-setup).
2. Familiarize yourself with the CodeCarbon library and its usage. Resources can be found in the [CodeCarbon Resources](#codecarbon-resources) section and in our in-class tutorial (Jan 22).
3. Implement your model of choice and run experiements to collect data.
4. Document your process and findings in a report.

### Models

| Model Name | Type | Architecture | Size | Documentation | Dataset | Pretrained Weights | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | NLP | Transformer | 116M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/bert) | [Synthetic Dataset from MilaBench](https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/bench/synth.py) | [HuggingFace BERT Model Card](https://huggingface.co/google-bert/bert-base-uncased) | {this pretrained model is 0.1B and it's from huggingface. milabench uses the dataset is synthetic for these [models](https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/prepare.py) but the huggingface model card also has the dataset it was pretrained on.} |
| Reformer | NLP | Transformer | 6M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/reformer) | [Synthetic Dataset from MilaBench](https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/bench/synth.py) | [HuggingFace Model](https://huggingface.co/docs/transformers/en/model_doc/reformer) | {ReformerConfig(). same as BERT for the dataset?} |
| T5 | NLP | Transformer | 0.2B | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/t5) | [Synthetic Dataset from MilaBench](https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/bench/synth.py) | [HuggingFace T5 Base Model Card](https://huggingface.co/google-t5/t5-base) | {same dataset as BERT?} |
| OPT | NLP | Transformer | 350M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/opt) | [TODO]() | [HuggingFace Opt-350M Model Card](https://huggingface.co/facebook/opt-350m) | {i am assuming its the same dataset as BERT cz its BERT adjacent models but also im not sure?} |
| Bart | NLP | Transformer | 0.1B | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/bart) | [TODO]() | [HuggingFace Bart Base Model Card](https://huggingface.co/facebook/bart-base) | {i am assuming its the same dataset as BERT cz its BERT adjacent models but also im not sure?} |
| BigBird | NLP | Transformer | ? | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/big_bird) | [TODO]() | [HuggingFace BigBird Roberta Base Model Card](https://huggingface.co/google/bigbird-roberta-base) | {milabench is using BigBirdConfig(attention_type="block_sparse"). i am assuming its the same dataset as BERT cz its BERT adjacent models but also im not sure?} |
| Albert | NLP | Transformer | 11.8M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/albert) | [TODO]() | [HuggingFace Albert Base V2 Model Card](https://huggingface.co/albert/albert-base-v2) | {i am assuming its the same dataset as BERT cz its BERT adjacent models but also im not sure?} |
| DistilBERT | NLP | Transformer | 67M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/distilbert) | [TODO]() | [HuggingFace DistilBERT Base Uncased Model Card](https://huggingface.co/docs/transformers/en/model_doc/distilbert) | {i am assuming its the same dataset as BERT cz its BERT adjacent models but also im not sure?} |
| Longformer | NLP | Transformer | ? | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/longformer) | [TODO]() | [HuggingFace Longformer Base 4096 Model Card](https://huggingface.co/allenai/longformer-base-4096) | {i am assuming its the same dataset as BERT cz its BERT adjacent models but also im not sure?} |
| Llava | MultiModal (NLP/CV) | Transformer | ? | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/llava) | [HuggingFace The Cauldron Dataset](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | [HuggingFace <MODEL> Model Card]() | {i couldnt find a pretrained model small enough? milabench is using the llava-hf/llava-1.5-7b-hf model} |
| Whisper | ASR | Transformer | 37.8M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/whisper) | [Synthetic Dataset from MilaBench](https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/bench/synth.py) | [HuggingFace Whisper Tiny Model Card](https://huggingface.co/openai/whisper-tiny) | {same as BERT for dataset?} |
| Dinov2 | ViT | Transformer | 0.3B | [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/dinov2) | [FakeImageNet](https://huggingface.co/datasets/InfImagine/FakeImageDataset) | [HuggingFace Dinov2 Large Model Card](https://huggingface.co/facebook/dinov2-large) | {the model file uses this [dataset](https://huggingface.co/datasets/helenlu/ade20k) but the table of the models you sent me has the FakeImageNet?} |
| V-Jepa2 | CV | Transformer | 632M | [HuggingFace Documentation](https://huggingface.co/docs/transformers/main/model_doc/vjepa2) | [MilaBench FakeVideo Dataset Generation](https://github.com/mila-iqia/milabench/blob/master/benchmarks/vjepa/prepare.py) | [HuggingFace V-JEPA2 Model Card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | {the dataset is generated by milabench i think} |
| ResNet50 | CV | CNN | 26M | [Pytorch Model Documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#resnet50) | [FakeImageNet](https://huggingface.co/datasets/InfImagine/FakeImageDataset) | [TODO]() | {does pytorch models have a model card? i put the model config page in the documentation for now.} |
| Resnet152 | CV | CNN | 60M | [Pytorch Model Documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet152.html#resnet152) | [FakeImageNet](https://huggingface.co/datasets/InfImagine/FakeImageDataset) | [TODO]() | {does pytorch models have a model card? i put the model config page in the documentation for now.} |
| ConvNext Large | CV | CNN | 200M | [Pytorch Model Documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_large.html#convnext-large) | [FakeImageNet](https://huggingface.co/datasets/InfImagine/FakeImageDataset) | [TODO]() | {does pytorch models have a model card? i put the model config page in the documentation for now.} |
| RegNet Y 128GF | CV | CNN,RNN | 693M | [Pytorch Model Documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_128gf.html#regnet-y-128gf) | [FakeImageNet](https://huggingface.co/datasets/InfImagine/FakeImageDataset) | [TODO]() | {does pytorch models have a model card? i put the model config page in the documentation for now.} |
| ViT-g/14 | CV | Transformer | 1B | [TODO]() | [FakeImageNet](https://huggingface.co/datasets/InfImagine/FakeImageDataset) | [TODO]() | {im a little confused by this one but here is the [huggingface dinov2-giant model](https://huggingface.co/facebook/dinov2-giant). the paper says its dinov2-giant-gpus} |
| PNA | Graphs | GNN | 4M | [TODO]() | [PCQM4Mv2](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.PCQM4Mv2.html) | [TODO]() | {[theres a link to the paper where this model is spawned from](https://github.com/mila-iqia/milabench/blob/master/benchmarks/geo_gnn/bench/models.py): [paper](https://arxiv.org/pdf/2004.05718). the dataset used seems to be a [subset](https://github.com/mila-iqia/milabench/blob/master/benchmarks/geo_gnn/pcqm4m_subset.py)} |
| DimeNet | Graphs | GNN | 500K | [TODO]() | [PCQM4Mv2](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.PCQM4Mv2.html) | [TODO]() | {[theres a link to the paper where this model is spawned from](https://github.com/mila-iqia/milabench/blob/master/benchmarks/geo_gnn/bench/models.py): [paper](https://arxiv.org/pdf/2003.03123). the dataset used seems to be a [subset](https://github.com/mila-iqia/milabench/blob/master/benchmarks/geo_gnn/pcqm4m_subset.py)} |
| GFlowNet | Graphs | GFlowNet, T. | 600M | [TODO]() | [TODO]() | [TODO]() | {[this is the paper about the model](https://arxiv.org/pdf/2106.04399) and [this is the github for the model implementation](https://github.com/GFNOrg/gflownet). they talk about a molecule dataset synthetically generated, are we using that as well? i did not put any model info for this one but they get the model from the github linked here and all other info are [here](https://github.com/mila-iqia/milabench/tree/master/benchmarks/recursiongfn)} |
| ? | ? | ? | ? | [TODO]() | [TODO]() | [TODO]() |

---

## Repository Structure
```
COMP597-starter-code
.
├── README.md 
├── config                              # Bash configurations to run the code
├── launch.py                           # Entrypoint to run training experiments
├── requirements.txt
├── scripts
│   └── env_setup.sh                    # Script to setup local conda environment
└── src
    ├── auto_discovery 
    ├── config                          # Configurations objects used to structure program inputs
    │   ├── config.py                   # Root config class containing all sub-configs
    │   ├── data                        # Sub-configs to load datasets
    │   ├── models                      # Sub-configs for models
    │   ├── trainer_stats               # Sub-configs to configure data measurements
    │   ├── trainers                    # Sub-configs to configure the trainers
    │   └── util                        # Utilities to create config objects
    ├── data                            # Module to load datasets
    │   ├── dataset                     # Thin wrapper around HF datasets module
    │   └── ...
    ├── models                          # Module to create models
    │   ├── gpt2                        # GPT2 example
    │   └── ...
    └── trainer                         # Module providing trainer classes that can train models
        ├── base.py                     # Abstract class defining the interface for trainer classes
        ├── simple.py                   # Simple 
        ├── ...
        └── stats                       # Module providing measurements classes for trainers
            ├── base.py                 # Abstract class defining the interface for measurement classes
            └── ...
```


### CodeCarbon Resources
- [CodeCarbon Colab Tutorial](https://colab.research.google.com/drive/1eBLk-Fne8YCzuwVLiyLU8w0wNsrfh3xq)

---

## GPT2 example
### How to run the codebase locally
1. Always activate the environment first using `source local_env.sh` or `. local_env.sh` or the commands provided in the [Environment Setup](#environment-setup) section if it is the first time.
2. To train a model, use the `launch.py` script with appropriate command-line arguments. 
    > List of command-line arguments can be found in the by running `python3 launch.py --help`. Alternatively, you can go through the config classes starting from `src/config/config.py`. Below are some of the high-level options.
    > - **Models (`--model`)**: the model to train. Currently supports "gpt2". Add the model you need to implement in the codebase.
    > - **Trainers (`--trainer`)**: the training method to use. Currently supports "simple". More trainers can be added as needed. 
    > - **Training Stats (`--trainer_stats`)**: the stats collection method to use during training. Currently supports "simple" and "codecarbon". More stats collection methods can be added as needed. TODO (ADD OR REMOVE TO BE DECIDED).
    > - **Dataset (`--data`)**: the data function to use to load the dataset. Currently, only `dataset` is available, thinly wraps some of the Hugging Face datasets library.
    > - **Batch Size (`--batch_size`)**: the batch size for training.
    > - **Learning Rate (`--learning_rate`)**: the learning rate for training. Adjust it based on the model and training setup as needed.

---

## Code documentation

This README only provided a high-level overview of the repository and what is can do. Please visit the more detailed [documentation](docs/ToC.md).
