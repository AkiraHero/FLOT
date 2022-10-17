

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
export PYTHONPATH="/mnt/petrelfs/juxiaoliang/project/petrel_utils:/mnt/petrelfs/juxiaoliang/project/sf/FLOT:${PYTHONPATH}"
export MASTER_ADDR="localhost"
export MASTER_PORT="${PORT}"

srun -p shlab_adg_s2 -N 1 --cpus-per-task=16 --quotatype reserved  --gres=gpu:4 python train.py --nb_iter 1 --dataset waymo --nb_points 8192 --batch_size 8 --nb_epochs 60  --distributed 1 --ngpus 4
