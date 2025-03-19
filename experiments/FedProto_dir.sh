python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_cifar10_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_cifar10_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_cifar100_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_cifar100_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_tumor_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_tumor_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_ich_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python main.py  --purpose ICH --device cuda:0 --global_seed 0 --use_wandb False --yamlfile .config/config_ich_conv2.yaml --strategy FedProto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
