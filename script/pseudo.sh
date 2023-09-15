python main.py --model convnext_base --pseudo true \
--nb_classes 600 \
--resume /workspace/UDA/ConvNeXt/pth/45.1_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /workspace/UDA/ConvNeXt/dataset_pseudo
