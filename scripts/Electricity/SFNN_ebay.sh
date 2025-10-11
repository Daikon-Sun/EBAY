model_name=SFNN
datapath=electricity
dataset=electricity

sl=168
rid=$1
mode=ebay

br=0
er=0.25

ep=15
lr=3e-6
adapt_iters=50

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
      --para_weight 0.3 \
      --para_weight2 0 \
      --beg_ratio $br \
      --end_ratio $er
done

# for pw in 0.1 0.3 1 3 10 ; do
#     for pl in 1 12 24 48 ; do
#         python3 -u run.py \
#           --root_path ./dataset/"$datapath"/ \
#           --data_path "$dataset".csv \
#           --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
#           --model $model_name \
#           --data $dataset \
#           --mode $mode \
#           --adapt_iters $adapt_iters \
#           --test_batch_size 168 \
#           --adapt_lr $lr \
#           --seq_len $sl \
#           --pred_len $pl \
#           --n_layers 3 \
#           --batch_size 32 \
#           --train_epochs $ep \
#           --weight_decay 0.00001 \
#           --dropout 0.5 \
#           --loss_fn MAE \
#           --learning_rate 0.0005 \
#           --para_weight $pw \
#           --para_weight2 0.005 \
#           --beg_ratio $br \
#           --end_ratio $er
#     done
# done
