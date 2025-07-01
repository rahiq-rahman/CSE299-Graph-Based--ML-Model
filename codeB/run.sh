MODEL=$1
DATASET=$2

# python ./codeP/main.py --dataset Cora --model gcn
python ./codeP/main.py --model "$MODEL" --dataset "$DATASET"
