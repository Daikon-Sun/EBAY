model_name=SFNN
datapath=solar
dataset=solar

sl=144
rid=$1
mode=ebay

br=0
er=0.25
bs=128

ep=15
lr=2e-4
adapt_iters=20

for pl in 1 12 24 48 ; do
    python3 -u run.py \
      --root_path ./dataset/"$datapath"/ \
      --data_path "$dataset".csv \
      --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
      --model $model_name \
      --data $dataset \
      --mode $mode \
      --adapt_iters $adapt_iters \
      --test_batch_size 144 \
      --adapt_lr $lr \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 2 \
      --need_norm \
      --batch_size $bs \
      --train_epochs $ep \
      --weight_decay 0 \
      --dropout 0 \
      --loss_fn MAE \
      --learning_rate 5e-4 \
      --min_lr 5e-5 \
      --para_weight 10 \
      --para_weight2 0 \
      --beg_ratio $br \
      --end_ratio $er
done

