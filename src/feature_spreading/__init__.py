from .spreading import SpreadingClassifier, SpreadingTransformer, knn_graph, normalized_adjacency, ppr_smooth, cluster_metric

__all__ = ['SpreadingClassifier', 'SpreadingTransformer', 'knn_graph', 'normalized_adjacency', 'ppr_smooth', 'cluster_metric']

# -- used by the `fs` command line tool and the scripts in experiments/ (not part of the library API)

DATA_DIR = "data"
MODEL    = "openai/clip-vit-large-patch14"

# short name -> huggingface dataset (all have `image` + `label` columns and a `train` split)
DATASETS = {
  "pets"          : "Multimodal-Fatima/OxfordPets_train",
  "cars"          : "Multimodal-Fatima/StanfordCars_train",
  "resisc45"      : "jonathan-roberts1/NWPU-RESISC45",
  "flowers"       : "nkirschi/oxford-flowers",
  "fashion_mnist" : "zalando-datasets/fashion_mnist",
  "food101"       : "ethz/food101",
}
