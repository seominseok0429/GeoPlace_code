python main.py \
	--eval true \
	--pseudo true \
	--pseudo_save_path /workspace/UDA/ConvNeXt/stage1_train/ \
	--is_target true \
        --model convnext_large \
        --data_set image_folder \
        --data_path /workspace/UDA/ConvNeXt/dataset/train2 \
        --eval_data_path /workspace/UDA/asia/train \
	--input_size 224 \
        --nb_classes 600 \
        --num_workers 8 \
        --warmup_epochs 0 \
	--resume ./warm_up_pth_L/checkpoint-7.pth \

rsync -av /workspace/UDA/usa/train/* stage1_train/.

python -m torch.distributed.launch --nproc_per_node=4 main.py --epochs 10 \
                --model convnext_large \
                --data_set image_folder \
                --batch_size 64 \
                --data_path /workspace/UDA/ConvNeXt/stage1_train/ \
                --eval_data_path /workspace/UDA/ConvNeXt/dataset/val \
                --nb_classes 600 \
                --num_workers 8 \
                --warmup_epochs 0 \
                --save_ckpt true \
                --output_dir ./stage1_pth \
                --finetune ./convnext_large_1k_384.pth \
                --cutmix 0 \
                --mixup 0 --lr 4e-4
