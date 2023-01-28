# reusable-nn-code

The Python package `reunn` contains **REU**sable code for defining, training and testing **N**eural **N**etworks. Facilitate your coding using object-oriented programing!

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