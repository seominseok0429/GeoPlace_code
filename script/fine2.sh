python main.py \
	--eval true \
        --model convnext_base \
        --data_set image_folder \
        --data_path /workspace/UDA/ConvNeXt/dataset/train2 \
        --eval_data_path /workspace/UDA/asia/train \
	--input_size 224 \
        --nb_classes 600 \
        --num_workers 8 \
        --warmup_epochs 0 \
	--resume /workspace/UDA/ConvNeXt/pth/57.78.pth \

