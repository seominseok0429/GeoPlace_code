# GeoPlace_code

------
```bash
# Warm-up
python -m torch.distributed.launch --nproc_per_node=4 main.py --epochs 10 \
                --model convnext_large \
                --data_set image_folder \
                --data_path /workspace/folder/ \
                --eval_data_path /workspace/UDA/geoplace/asia/test \
                --nb_classes 205 \
                --num_workers 8 \
                --warmup_epochs 0 \
                --save_ckpt true \
                --output_dir ./warm_up_pth_L/ \
                --finetune ./convnext_large_1k_384.pth \
                --cutmix 0 \
                --mixup 0 --lr 4e-4 \
```
