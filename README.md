#Torch

## Overview

*Torch* is a app for doing exactly two things:

1) Easily extracting the activations from every single intermediate operation in a PyTorch model—no
   modifications needed—in one line of code. "Every operation" means every operation; "one line" means one line.
2) Understanding the model's computational structure via an intuitive automatic visualization and extensive
   metadata  about the network's computational graph.

Here it is in action for a very simple recurrent model; as you can see, you just define the model like normal and pass
it in, and *TorchLens* returns a full log of the forward pass along with a visualization:

And here it is for a very complex transformer model with 1932 operations
in its forward pass; you can grab the saved outputs of every last one:

## How-To Guide

Below is a quick demo of how to use it; for an interactive demonstration.

The main function of *TorchLens* is `log_forward_pass`: when called on a model and input, it runs a
forward pass on the model and returns a ModelHistory object containing the intermediate layer activations and
accompanying metadata, along with a visual representation of every operation that occurred during the forward pass:

```python
import torch
import torchvision
import torcheer as tl

alexnet = torchvision.models.alexnet()
x = torch.rand(1, 3, 224, 224)
model_history = tl.log_forward_pass(alexnet, x, layers_to_save='all', vis_opt='unrolled')
print(model_history)

'''
Log of AlexNet forward pass:
	Model structure: purely feedforward, without branching; 23 total modules.
	24 tensors (4.8 MB) computed in forward pass; 24 tensors (4.8 MB) saved.
	16 parameter operations (61100840 params total; 248.7 MB).
	Random seed: 3210097511
	Time elapsed: 0.288s
	Module Hierarchy:
		features:
		    features.0, features.1, features.2, features.3, features.4, features.5, features.6, features.7, 
		    features.8, features.9, features.10, features.11, features.12
		avgpool
		classifier:
		    classifier.0, classifier.1, classifier.2, classifier.3, classifier.4, classifier.5, classifier.6
	Layers:
		0: input_1_0 
		1: conv2d_1_1 
		2: relu_1_2 
		3: maxpool2d_1_3 
		4: conv2d_2_4 
		5: relu_2_5 
		6: maxpool2d_2_6 
		7: conv2d_3_7 
		8: relu_3_8 
		9: conv2d_4_9 
'''
```
