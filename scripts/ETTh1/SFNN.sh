model_name=SFNN
datapath=ETT-small
dataset=ETTh1

for rid in {0..9} ; do
    for sl in 336 168 336 672 1344 ; do
        for pl in 720 96 192 336 720 ; do
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 1 \
              --need_norm --layernorm \
              --batch_size 64 \
              --train_epochs 50 \
              --weight_decay 0.001 \
              --dropout 0.7 \
              --loss_fn MAE \
              --learning_rate 0.0005 \
              --norm_len 96
            exit
        done
    done
done
