# feature_spreading

- Very simple, lightweight, low-shot (transductive, for now) semi-supervised learning technique.
- In contrast to label propagation (e.g. in `sklearn` or Eq 6 of [this paper](https://arxiv.org/pdf/1904.04717.pdf)), this propagates the _features_ before training / inference.
  - This code just uses the raw features, which seems fine if we have a good featurizer like CLIP.
  - Sketch of jointly learning the features is [here](https://github.com/bkj/ez_ppnp/blob/cas/main.py#L115)
- Experiments use image features extracted in [ez_feat](https://github.com/jataware/ez_feat) format

