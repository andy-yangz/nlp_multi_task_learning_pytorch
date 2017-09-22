echo "Running Model"

# echo "POS"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 2 \
#         --nchunk_layers 0 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --bi \
#         --train_mode 'POS' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/pos_model'

# echo "Chunk"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 0 \
#         --nchunk_layers 2 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --train_mode 'Chunk' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/chunk_model'

# echo "Joint Training POS 3 Chunk 3"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 3 \
#         --nchunk_layers 3 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --train_mode 'Joint' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/joint_same'

# echo "Joint Training POS 1 Chunk 2"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 1 \
#         --nchunk_layers 2 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --bi \
#         --train_mode 'Joint' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/joint_diff'

# echo "Joint Training POS 2 Chunk 3"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 2 \
#         --nchunk_layers 3 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --train_mode 'Joint' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/joint_diff'

# echo "Joint Training POS 1 Chunk 2"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 1 \
#         --nchunk_layers 2 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --train_mode 'Joint' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/joint_diff'

# echo "Joint Training POS 2 Chunk 1"
# python main.py --data './data' \
#         --emsize 256 \
#         --npos_layers 2 \
#         --nchunk_layers 1 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 10 \
#         --cuda \
#         --train_mode 'Joint' \
#         --epochs 300 \
#         --log_interval 20 \
#         --save './result/joint_diff'



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

echo "Number of Layers"
for nlayers in 1 3 
do
    python main.py --data './data' \
            --emsize 256 \
            --npos_layers $nlayers \
            --nchunk_layers 0 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'POS' \
            --epochs 300 \
            --log_interval 20 \
            --save './result/pos_model'
done

for nlayers in 1 3 
do
    python main.py --data './data' \
            --emsize 256 \
            --npos_layers 0 \
            --nchunk_layers $nlayers \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'Chunk' \
            --epochs 300 \
            --log_interval 20 \
            --save './result/chunk_model'
done

echo "Number of hidden units"
for nhid in 256 512
do
    python main.py --data './data' \
            --emsize 256 \
            --npos_layers 1 \
            --nchunk_layers 0 \
            --nhid $nhid \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'POS' \
            --epochs 300 \
            --log_interval 20 \
            --save './result/pos_model'
done

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

for rnn_type in 'GRU' 'Elman'
do 
    python main.py --data './data' \
            --emsize 256 \
            --npos_layers 1 \
            --nchunk_layers 0 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --rnn_type $rnn_type \
            --cuda \
            --train_mode 'POS' \
            --epochs 300 \
            --log_interval 20 \
            --save './result/pos_model'
done

for dropout in 0.1 0.4 0.6
do
    python main.py --data './data' \
            --emsize 256 \
            --npos_layers 1 \
            --nchunk_layers 0 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --dropout $dropout \
            --cuda \
            --train_mode 'POS' \
            --epochs 300 \
            --log_interval 20 \
            --save './result/pos_model'
done