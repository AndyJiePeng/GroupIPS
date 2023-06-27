for size in 2000 4000 8000 16000 32000
do
    python3 main.py \
    --t_d 100 \
    --group_k 20 \
    --size "$size" \
    --beta -1 \
    --truePropensity True \
    --treatmentEffect 0.8 \
    --gpu 7 \
    --epislon 0.2 \
    --seed 618 \
    --epoch 500 \
    --lr 0.003 \
    --filePath "num_sample_k100" &
done