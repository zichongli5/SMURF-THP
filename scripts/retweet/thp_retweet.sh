device=1
data=../data/data_retweet/
batch=16 # batch size
n_head=3 # number of heads of transformer encoder
n_layers=3 # number of layers of transformer encoder
d_model=64 # dimension of the transformer model
d_rnn=64 # dimension of RNN layer (not used now)
d_inner=256 # dimension of the hidden layer in transformer model(Feed Forward Network)
d_k=16 
d_v=16
optimizer=adam # optimizer
scheduler=cosLR # type of lr scheduler; 'steplr' for StepLR, 'reduce' for ReduceLROnPlateau
dropout=0.1
lr=5e-3
smooth=0.1 # smoothing of loss used for type prediction (0.1 is used in THP)
epoch=40 
eval_epoch=41 # number of epoch we want to start to evaluate
normalize=normal

## langevin sampling parameter
langevin_step=5e-5 # sampling step size; decrease for better sampling, but cost more step
n_steps=1000 # sampling step count; increase for better sampling, but cost more time
n_save_steps=100 # save step per n_save_steps
n_samples=100 # number of generated sample; increase for more sample, but cost more memory
sampling_method=normal

save_path=./checkpoints/retweet/
save_name=retweet_thp_model_best_${normalize}

eval_quantile=-1
eval_quantile_step=0.05

cd ../..
model=thp

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py -normalize $normalize -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -save_name $save_name -save_path $save_path

load_path_name=./checkpoints/retweet/retweet_thp_model_best_${normalize}.pth
save_result=./results/retweet/retweet_thp_sample


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py -sampling_method $sampling_method -save_result $save_result -normalize $normalize -just_eval -n_steps $n_steps -n_save_steps $n_save_steps -langevin_step $langevin_step -n_samples $n_samples -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -load_path_name $load_path_name
