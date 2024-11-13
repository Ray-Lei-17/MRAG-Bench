#!/bin/bash
#SBATCH --job-name=rag             # 作业名称-请修改为自己任务名字 
#SBATCH --output=/home/siqilei2/MRAG-Bench/output/output_%j.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=/home/siqilei2/MRAG-Bench/error/error_%j.txt          # 标准错误文件名-请修改为自己路径
#SBATCH --cpus-per-task=4             # 每个任务使用的CPU核心数
#SBATCH --mem=100G                      # 申请100GB内存
#SBATCH --time=12:00:00               # 运行时间限制，格式为hh:mm:ss
conda init
conda activate ray-rag #-请修改为自己conda环境
# CUDA_VISIBLE_DEVICES="0,1" python3 eval/models/test.py #-请修改为指定运行的gpu-id，设置为0或1或2或3。

# export HF_HOME="./cache"
# export HF_HUB_CACHE="./cache"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1

array=($(echo $1 | grep -o .))
# echo "${array[@]}"
# echo "results_json/$1.jsonl"

python3 eval/models/llava_one_vision.py \
    --answers-file "results_json/$1.jsonl" \
    --use-rag True \
    --use-retrieved-examples True \
    --use-mix-examples False \
    --retrieval-order "${array[@]}"