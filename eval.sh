#vim

#torchrun --nproc_per_node=4 --master_port=15400 eval_linear_vim.py --arch vim_tiny_patch16_224 --pretrained_weights ../checkpoint/Vim/Tiny/checkpoint.pth --data_path ./dataset/ImageNet/ --output_dir ../checkpoint/Vim/Tiny/ --n_last_blocks 8 --evaluate

# torchrun --nproc_per_node=4 --master_port=15400 eval_linear_vim.py --arch vim_small_patch16_224 --wd 0.001 --pretrained_weights ../checkpoint/Vim/Small/checkpoint.pth --data_path ./dataset/ImageNet/ --output_dir ../checkpoint/Vim/Small/ --n_last_blocks 4 --evaluate

# torchrun --nproc_per_node=4 --master_port=15400 eval_linear_vim.py --arch vim_base_patch16_224 --wd 0.01 --pretrained_weights ../checkpoint/Vim/Base/checkpoint.pth --data_path ./dataset/ImageNet/ --output_dir ../checkpoint/Vim/Base/ --n_last_blocks 8 --evaluate

#mambar
# torchrun --nproc_per_node=4 --master_port=15400 eval_linear_mambar.py --arch mambar_base_patch16_224 --wd 0.01 --pretrained_weights ../checkpoint/Mambar/Base/checkpoint.pth --data_path ./dataset/ImageNet/ --output_dir ../checkpoint/Mambar/Base/ --n_last_blocks 8 --evaluate
