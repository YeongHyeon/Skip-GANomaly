Skip-GANomaly
=====

Implementation of Skip-GANomaly with MNIST dataset [<a href="https://github.com/YeongHyeon/GANomaly">Related repository</a>].

## Architecture
<div align="center">
  <img src="./figures/skipganomaly.png" width="500">  
  <p>Simplified Skip-GANomaly architecture.</p>
</div>

## Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="800">  
  <p>Graph of Skip-GANomaly.</p>
</div>

## Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="600">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by Skip-GANomaly.</p>
</div>

<div align="center">
  <img src="./figures/test-box.png" width="400">
  <p>Box plot with encoding loss of test procedure.</p>
</div>

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] S Akcay, et al. (2018). <a href="https://arxiv.org/abs/1901.08954">Skip-ganomaly: Skip connected and adversarially trained encoder-decoder anomaly detection.</a>. arXiv preprint arXiv:1901.08954.
