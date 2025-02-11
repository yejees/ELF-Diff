export CUDA_VISIBLE_DEVICES=6

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 320 --learn_sigma True --noise_schedule cosine --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True" 

python test.py \
 $MODEL_FLAGS \
 --model_path ./results/ema_0.9999_500000.pt \
 --timestep_respacing 50 \
 --sample_method ELF \
 --save_dir ./sample_results/results

