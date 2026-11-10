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
