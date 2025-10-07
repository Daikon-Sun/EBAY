model_name=SFNN
datapath=ETT-small
dataset=ETTh2

for rid in {0..9} ; do
    for sl in 168 336 672 1344 ; do
        for pl in 96 192 336 720 ; do
            if [ $pl -eq 96 ] ; then
                mixer="--mixer"
            elif [ $pl -eq 192 ] ; then
                mixer="--mixer"
            elif [ $pl -eq 336 ] ; then
                mixer=""
            elif [ $pl -eq 720 ] ; then
                mixer=""
            fi
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 1 \
              $mixer \
              --need_norm --layernorm \
              --batch_size 64 \
              --train_epochs 100 \
              --weight_decay 0.0015 \
              --dropout 0.7 \
              --loss_fn MAE \
              --learning_rate 0.0005 \
              --norm_len 0
        done
    done
done
