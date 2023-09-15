python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model convnext_base --drop_path 0.1 \
--data_set image_folder \
--nb_classes 600 \
--batch_size 64 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /workspace/UDA/ConvNeXt/dataset/train \
--eval_data_path /workspace/UDA/asia/test \
--output_dir ./.
