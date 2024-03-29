# reusable-nn-code

**Deprecated**: This repository will ⚠️ no longer be updated or maintained⚠️, but the existing contents will remain available!
We strongly recommend you to use official high-level interfaces, like [pytorch-ignite](https://github.com/pytorch/ignite).

The Python package `reunn` contains **REU**sable code for defining, training and testing **N**eural **N**etworks. Facilitate your coding using object-oriented programing!

`TaskPipeline` in this package is particularly suitable for writing unit tests or solving simple ML tasks. Now, it can deal with:

* supervised **regression** tasks: `SupervisedClassificationTaskPipeline`
* supervised **classification** tasks: `SupervisedClassificationTaskPipeline`.

However, for complex tasks, we suggest that you should write your pipeline manually!

`NetStats` provides statistical information about your neural network, like:

* the number of multiply-accumulate operations (MACs) required during computation
* the total number of parameters in the network

## Installation

### Install from Source Code

From [GitHub](https://github.com/AllenYolk/reusable-nn-code):

```shell
git clone https://github.com/AllenYolk/reusable-nn-code.git
cd reusable-nn-code
pip install .
```

## TODO:

* [x] Add `setup.py` and write the installation guide.
* [x] Implement `TaskPipeline`. Use the "Bridge" design pattern here.
* [x] Implement `NetStats`. Use the "Bridge" design pattern here. Use `fvcore` to implement pytorch-based `NetStats`.
* [x] Add silent training mode.
* [x] Fix bug: when training with `validation=False`, `validation_loss` is `None`. But it is accessed and compared with `min_loss`.
* [ ] Add continual learning task pipeline.
* [x] Add [`spikingjelly`](https://github.com/fangwei123456/spikingjelly) implementation (for SNN task pipelines and network statistics).
* [x] Add hyperparameter to task pipelines.
* [x] Close the summary writer when destructing the task pipeline.
* [x] Add support for self-defined implementation.
