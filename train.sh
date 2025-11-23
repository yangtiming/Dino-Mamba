##Vim
#torchrun --nproc_per_node=4 --master_port=15100 main_dino_vim.py --arch vim_tiny_patch16_224 --data_path /cis/home/gwei10/dataset/ImageNet/train --output_dir ./output_vim_tiny/

##Mambar
#torchrun --nproc_per_node=4 --master_port=15100 main_dino_mambar.py --arch mambar_tiny_patch16_224 --data_path /cis/home/gwei10/dataset/ImageNet/train --output_dir ./output_mambar/
