model_name=SFNN
datapath=weather
dataset=weather

sl=96
rid=$1
mode=freezed

br=0
er=0.25

ep=20
lr=1e-5
adapt_iters=3

for pl in 24 ; do
    python3 -u run.py \
      --root_path ./dataset/"$datapath"/ \
      --data_path "$dataset".csv \
      --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
      --model $model_name \
      --mode $mode \
      --adapt_iters $adapt_iters \
      --test_batch_size 10 \
      --adapt_lr $lr \
      --data $dataset \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 2 \
      --mixer \
      --need_norm --layernorm \
      --batch_size 128 \
      --train_epochs 30 \
      --weight_decay 0.0000 \
      --dropout 0.0 \
      --loss_fn MAE \
      --learning_rate 0.001 \
      --para_weight 10 \
      --beg_ratio $br \
      --end_ratio $er
done
