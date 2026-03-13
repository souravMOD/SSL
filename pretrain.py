import os

import lightly_train
from ultralytics import settings

my_transform_args = {
    "random_resize": {
        "min_scale": 0.1,
    },
    "image_size": (640, 640),
    "color_jitter": None,
}

"""DINO Pretraining with LightlyTrain"""
# "global_view_1": {                     # modifications for second global view (cannot be disabled)
#     "gaussian_blur": {                 # can be disabled by setting to None
#         "prob": float,                 
#         "sigmas": tuple[float, float],
#         "blur_limit": int | tuple[int, int]
#     },
#     "solarize": {                      # can be disabled by setting to None
#         "prob": float,
#         "threshold": float
#     }
# },
# "local_view": {                        # configuration for local views (can be disabled by setting to None)
#     "num_views": int,                  # number of local views to generate
#     "view_size": tuple[int, int],      # size of local views
#     "random_resize": {                 # can be disabled by setting to None
#         "min_scale": float,
#         "max_scale": float
#     },
#     "gaussian_blur": {                 # can be disabled by setting to None
#         "prob": float,
#         "sigmas": tuple[float, float],
#         "blur_limit": int | tuple[int, int]
#     }
# }

data_path = os.environ.get("SSL_DATA_PATH", "/data")
output_dir = os.environ.get("SSL_OUTPUT_DIR", "out/my_experiment")
epochs = int(os.environ.get("SSL_EPOCHS", "100"))
batch_size = int(os.environ.get("SSL_BATCH_SIZE", "4"))

if __name__ == "__main__":
    lightly_train.train(
        out=output_dir,
        model="ultralytics/yolo11l.yaml",
        data=data_path,
        method="distillation",
        epochs=epochs,
        transform_args=my_transform_args,
        batch_size=batch_size,
    )