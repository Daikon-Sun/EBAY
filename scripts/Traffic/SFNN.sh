model_name=SFNN
datapath=traffic
dataset=traffic

sl=336
pl=1
rid=$1
for br in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 ; do
    # er = br + 0.2
    er=$(python -c "print($br + 0.2)")
    echo $br $er
    python -u run.py \
      --root_path ./dataset/"$datapath"/ \
      --data_path "$dataset".csv \
      --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
      --model $model_name \
      --data $dataset \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 2 \
      --batch_size 16 \
      --train_epochs 150 \
      --weight_decay 0 \
      --dropout 0 \
      --loss_fn MSE \
      --learning_rate 0.0005 \
      --min_lr 1e-5 \
      --beg_ratio $br \
      --end_ratio $er
done
