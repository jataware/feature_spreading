# feature_spreading

__Scratch experimental code__

- Very simple, lightweight, low-shot (transductive, for now) semi-supervised learning technique.
- In contrast to label propagation (e.g. in `sklearn` or Eq 6 of [this paper](https://arxiv.org/pdf/1904.04717.pdf)), this propagates the _features_ before training / inference.
  - This code just uses the raw features, which seems fine if we have a good featurizer like CLIP.
  - Sketch of jointly learning the features is [here](https://github.com/bkj/ez_ppnp/blob/cas/main.py#L115)
- Experiments use CLIP (`openai/clip-vit-large-patch14`) image features

## Usage

```bash
pixi install
pixi run ./run.sh            # all datasets
pixi run ./run.sh pets cars  # subset
```

`fs download <dataset>` pulls the images from Hugging Face, extracts CLIP features for a random subset (`--n_obs`, default 10k) and saves them to `data/<dataset>/<model>/{X,y}.npy`.
`fs run <dataset>` runs the 1-shot LinearSVC baseline vs. PPR feature spreading on those features.

Datasets: `pets`, `cars`, `resisc45`, `flowers`, `fashion_mnist`, `food101` (see `src/feature_spreading/__init__.py` for HF ids).

