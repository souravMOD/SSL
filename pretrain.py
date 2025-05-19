import lightly_train
from ultralytics import settings

data_path = r"D:\DATASET\self_supervised_pretrain\GenV2\train\images"
if __name__ == "__main__":
    # Pretrain with LightlyTrain.
    lightly_train.train(
        out="out/my_experiment",            # Output directory.
        model="ultralytics/yolo11l.yaml",   # Pass the YOLO model.
        data=data_path, 
        method="distillation",                   # Path to a directory with training images.
        epochs=100,                         # Adjust epochs for faster training.
        batch_size=64,                      # Adjust batch size based on hardware.
    )