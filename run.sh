echo "Running Model"

echo "POS tagging"
python main.py --data './data' \
        --emsize 256 \
        --nlayers 1 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 10 \
        --cuda \
        --train_mode 'POS' \
        --test_times 10 \
        --epochs 500 \
        --log_interval 20 \
        --save './result/pos_model'

echo "Chunking"
python main.py --data './data' \
        --emsize 256 \
        --nlayers 1 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 10 \
        --cuda \
        --train_mode 'Chunk' \
        --test_times 10 \
        --epochs 500 \
        --log_interval 20 \
        --save './result/chunk_model'

# echo "Embedding size"
# for emsize in 128 256 512 
# do
#     echo "Embedding size $emsize"
#     python main.py --data './data' \
#             --emsize $emsize \
#             --nlayers 1 \
#             --nhid 128 \
#             --batch_size 128 \
#             --seq_len 15 \
#             --cuda \
#             --epochs 300 \
#             --log_interval 20 \
#             --save './result/pos_model'
# done

# echo "Number of Layers"
# for nlayers in 2 3 
# do
#     echo "NUmber of layers $nlayers"
#     python main.py --data './data' \
#                 --emsize 128 \
#                 --nlayers $nlayers \
#                 --nhid 128 \
#                 --batch_size 128 \
#                 --seq_len 15 \
#                 --cuda \
#                 --epochs 300 \
#                 --log_interval 20 \
#                 --save './result/pos_model'
# done

# echo "Number of hidden units"
# for nhid in 256 512
# do
#     echo "Number of hidden units $nhid"
#     python main.py --data './data' \
#                 --emsize 128 \
#                 --nlayers 1 \
#                 --nhid $nhid \
#                 --batch_size 128 \
#                 --seq_len 15 \
#                 --train_mode 'POS' \
#                 --cuda \
#                 --epochs 300 \
#                 --log_interval 10 \
#                 --save './result/pos_model'
# done

# echo "Sequence Length"
# for seq_len in 10 20
# do
#     python main.py --data './data' \
#             --emsize 128 \
#             --nlayers 1 \
#             --nhid 128 \
#             --batch_size 128 \
#             --seq_len $seq_len \
#             --train_mode 'POS' \
#             --cuda \
#             --epochs 300 \
#             --log_interval 10 \
#             --save './result/pos_model'
# done

# for dropout in 0.4 0.6
# do
#     python main.py --data './data' \
#             --emsize 128 \
#             --nlayers 1 \
#             --nhid 128 \
#             --batch_size 128 \
#             --seq_len 15 \
#             --dropout $dropout \
#             --train_mode 'POS' \
#             --cuda \
#             --epochs 300 \
#             --log_interval 10 \
#             --save './result/pos_model'
# done

# for rnn_type in 'GRU' 'Elman'
# do 
#     python main.py --data './data' \
#             --emsize 128 \
#             --nlayers 1 \
#             --nhid 128 \
#             --batch_size 128 \
#             --seq_len 15 \
#             --rnn_type $rnn_type \
#             --train_mode 'POS' \
#             --cuda \
#             --epochs 300 \
#             --log_interval 10 \
#             --save './result/pos_model'
# done

