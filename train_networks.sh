# MNIST DNN Lipschitz training (toy) with 8 hidden layers
python main.py --mode train --model Toy --gamma 1 --layer Sandwich --scale small  --dataset mnist --epochs 200
python main.py --mode train --model Toy --gamma 1 --layer Orthogon --scale small --dataset mnist --epochs 200 
python main.py --mode train --model Toy --gamma 1 --layer Aol --scale small --dataset mnist --epochs 200
python main.py --mode train --model Toy --gamma 1 --layer SLL --scale small --dataset mnist --epochs 200

