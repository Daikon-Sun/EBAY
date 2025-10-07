model_name=SFNN
datapath=electricity
dataset=electricity

sl=168
rid=$1
mode=retrain

br=0
er=0.25
bs=1024

ep=25
lr=3e-3
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
      --test_batch_size 168 \
      --adapt_lr $lr \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 3 \
      --batch_size 32 \
      --train_epochs $ep \
      --weight_decay 0.00001 \
      --dropout 0.5 \
      --loss_fn MAE \
      --learning_rate 0.0005 \
      --beg_ratio $br \
      --end_ratio $er
done
