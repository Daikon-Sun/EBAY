model_name=SFNN
datapath=traffic
dataset=traffic

sl=168
rid=$1
mode=ebay

rid=$1
br=0
er=0.25

ep=15
lr=5e-5
adapt_iters=10

for pl in 1 12 24 48 ; do
    python3 -u run.py \
      --root_path ./dataset/"$datapath"/ \
      --data_path "$dataset".csv \
      --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
      --model $model_name \
      --data $dataset \
      --mode $mode \
      --adapt_iters $adapt_iters \
      --adapt_lr $lr \
      --test_batch_size 168 \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 2 \
      --batch_size 32 \
      --train_epochs $ep \
      --weight_decay 0.0000 \
      --dropout 0. \
      --loss_fn MSE \
      --learning_rate 0.0004 \
      --min_lr 1e-5 \
      --beg_ratio $br \
      --end_ratio $er \
      --para_weight 0 \
      --para_weight2 5
done
