model_name=SFNN
datapath=ETT-small
dataset=ETTm1

for rid in {0..9} ; do
    for sl in 96 192 384 672 1344 ; do
        for pl in 96 192 336 720 ; do
            if [ $pl -eq 96 ] ; then
                wd=0.0003
            elif [ $pl -eq 192 ] ; then
                wd=0.0006
            elif [ $pl -eq 336 ] ; then
                wd=0.0009
            elif [ $pl -eq 720 ] ; then
                wd=0.0012
            fi
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 2 \
              --need_norm --layernorm \
              --norm_len 0 \
              --batch_size 256 \
              --train_epochs 200 \
              --weight_decay $wd \
              --dropout 0.7 \
              --loss_fn MAE \
              --learning_rate 0.0005
        done
    done
done
