model_name=SFNN
datapath=ETT-small
dataset=ETTh2

sl=168
rid=$1
mode=ebay

br=0
er=0.25

ep=20
lr=1e-5
adapt_iters=10

# for pl in 48 1 12 24 48 ; do
for pl in 24 ; do
    python3 -u run.py \
      --root_path ./dataset/"$datapath"/ \
      --data_path "$dataset".csv \
      --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
      --model $model_name \
      --data $dataset \
      --mode $mode \
      --adapt_iters $adapt_iters \
      --test_batch_size 168 \
      --adapt_lr $lr \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 1 \
      --mixer \
      --need_norm --layernorm \
      --batch_size 64 \
      --train_epochs 100 \
      --weight_decay 0.0015 \
      --dropout 0.7 \
      --loss_fn MAE \
      --learning_rate 0.0005 \
      --norm_len 0 \
      --para_weight 10 \
      --beg_ratio $br \
      --end_ratio $er
done
