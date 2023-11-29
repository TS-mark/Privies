import os

def generate_filepath():
    base_dir = "Privies"
    sub_dirs = [
        "data/dataloader",
        "data/augment",
        "model/head/classify",
        "model/head/object detect",
        "model/head/segment",
        "model/backbone",
        "loss",
        "val",
        "train_manager",
        "res"
    ]
    
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)

generate_filepath()
