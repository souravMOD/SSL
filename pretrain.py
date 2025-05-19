import lightly_train
from ultralytics import settings




my_transform_args = {
    "random_resize": {
        "min_scale": 0.1
    },
    "image_size": (640, 640),
    "color_jitter": None,
}

'''DINO Pretraining with LightlyTrain'''
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

data_path = r"D:\DATASET\self_supervised_pretrain\GenV2\train\images"
if __name__ == "__main__":
    # Pretrain with LightlyTrain.
    lightly_train.train(
        out="out/my_experiment",            # Output directory.
        model="ultralytics/yolo11l.yaml",   # Pass the YOLO model.
        data=data_path, 
        method="distillation",                   # Path to a directory with training images.
        epochs=100,   
        transform_args= my_transform_args,                      # Adjust epochs for faster training.
        batch_size=4,                      # Adjust batch size based on hardware.
    )