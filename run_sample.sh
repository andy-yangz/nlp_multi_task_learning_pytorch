echo "Running Model on samples"
python main.py --data './data/sample' \
            --emsize 20 \
            --nlayers 1 \
            --nhid 20 --batch_size 2 --seq_len 10 --cuda --log_interval 2
