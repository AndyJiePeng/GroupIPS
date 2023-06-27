for action in  20 30 40 80 120 
do
    python3 main.py \
    --t_d "$action" \
    --group_k "$((action/5))" \
    --size 10000 \
    --beta -1 \
    --truePropensity True \
    --treatmentEffect 0.8 \
    --gpu 6 \
    --epislon 0.2 \
    --seed 618 \
    --epoch 500 \
    --lr 0.003 \
    --filePath "num_action" &
done