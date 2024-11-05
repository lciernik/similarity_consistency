# Representational similarity consistency across datasets

<p align="center">
  <img width="500" src="./figure1_v6.svg">

</p>

This repository contains the code and experiments for the paper 
*"Training objective drives the consistency of representational similarity across datasets"* ([arXiv](https://arxiv.org/abs/xxxx.xxxxx)).



## Table of Contents
1. [About](#about)
2. [Repository overview](#repository-overview)
3. [Installation](#installation)
4. [How to run](#how-to-run)


## About
*The Platonic Representation Hypothesis* claims that recent foundation models are converging to a shared representation 
space as a function of their downstream task performance, irrespective of the objectives and data modalities used to 
train these models . Representational similarity is generally measured for individual datasets 
and is not necessarily consistent across datasets. Thus, one may wonder whether this convergence of model representations 
is confounded by the datasets commonly used in machine learning. Here, we propose a systematic way to measure how 
representational similarity between models varies with the set of stimuli used to construct the representations. 
We find that the objective function is the most crucial factor in determining the consistency of representational 
similarities across datasets. Specifically, self-supervised vision models, but not image classification or image-text 
models, learn representations whose relative pairwise similarities generalize from one dataset to another.
Moreover, the correspondence between representational similarities and the models' task behavior varies by dataset type, 
being most strongly pronounced for single-domain datasets.  Our work provides a framework for systematically measuring 
similarities of model representations across datasets and linking those similarities to differences in task behavior.

## Repository and project overview

Brief description of the main components in your repository.

## Installation

1. Clone the repository
2. Install dependencies
3. Configure environment variables (if needed)

## How to Run

### Datasets
Instructions for preparing or obtaining the necessary data.

### Feature Extraction

### Model Similarities
#### Dataset subsampling
##### ImageNet-1k
##### Other wds_datasets**

#### Model similarity computation

### Linear Probing (Single model downstream task evaluation)
Instructions for evaluating the model and reproducing results.

### How to reproduce our results


## Acknowledgements

- Funding sources
- Collaborators and contributors
- Other resources or codebases used
- thingvision 
- CLIP benchmark

## License

This project is licensed under the [LICENSE NAME] - see the LICENSE file for details.

## Contact & Citation

- Your Name
- Project Link

If you find this code useful in your research, please cite our paper:

```bibtex
@article{author2024title,
    title={Paper Title},
    author={Author, First and Second, Author},
    journal={Journal Name},
    year={2024}
}
```
