# CurML: A Curriculum Machine Learning Library

A library & toolkit for Curriculum Learning.

Actively under development by @THUMNLab

## Target

1. To reproduce, evaluate and compare existing Curriculum Learning algorithms

2. To provide baselines and benchmarks for Curriculum Learning study.

3. To utilize Curriculum Learning as a plugin to improve the performance of existing works.

## Introduction

The figure below shows the overall framework of the library. The main part is a **CL Trainer** class, composed of a **CL Algorithm** class and a **Model Trainer** class, which interact with each other through five **APIs** of the CL Algorithm class to implement the curriculum.

<img src="./docs/img/framework.svg">

An general workflow of curriculum machine learning is illustrated below. 

<img src="./docs/img/flow.svg">

## Environment
1. python > 3.6

2. pytorch > 1.9.0

## Quick Start

``` bash
# 1. clone from the repository
git clone https://github.com/zhouyw16/CurML.git
cd CurML

# 2. pip install local module: curriculum
pip install -e .

# 3. run the example code
python examples/base.py
```

## License
We follow [Apache license](LICENSE) across the entire codebase from v0.2.