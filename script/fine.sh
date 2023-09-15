python -m torch.distributed.launch --nproc_per_node=4 main.py --epochs 10 \
                --model convnext_base \
                --data_set image_folder \
		--batch_size 128 \
                --data_path /workspace/UDA/ConvNeXt/stage1_train/ \
                --eval_data_path /workspace/UDA/ConvNeXt/dataset/val \
                --nb_classes 600 \
                --num_workers 8 \
                --warmup_epochs 0 \
                --save_ckpt true \
                --output_dir ./. \
                --finetune ./convnext_base_1k_384.pth \
                --mixup 0 --lr 4e-4

