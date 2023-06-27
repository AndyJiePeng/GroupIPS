for beta in -2 -1 -0.5 0
do
    python3 main.py \
    --t_d 100 \
    --group_k 20 \
    --size 10000 \
    --beta "$beta" \
    --truePropensity True \
    --treatmentEffect 0.8 \
    --gpu 5 \
    --epislon 0.2 \
    --seed 618 \
    --filePath "vary_beta" \
    --epoch 600 \
    --lr 0.001  &
done