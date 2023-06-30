## model parameter
device=3 # GPU device number
data=../data/data_mimic/fold1/ # datasets directory
batch=1 # batch size
decoder=encode
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
lr=1e-4
smooth=0.1 # smoothing of loss used for type prediction (0.1 is used in THP)
epoch=70
eval_epoch=71 # number of epoch we want to start to evaluate
normalize=normal
loss_lambda=3 

## langevin sampling parameter
langevin_step=3e-2 # sampling step size; decrease for better sampling, but cost more step
n_steps=300 # sampling step count; increase for better sampling, but cost more time
n_save_steps=100 # save step per n_save_steps
n_samples=100 # number of generated sample; increase for more sample, but cost more memory

## quantile of calibration score
eval_quantile=-1 # -1 stand for np.arange(eval_quantile_step, 1, eval_quantile_step); could also be a float in (0,1)
eval_quantile_step=0.05 # used when eval_quantile=-1

## parameter for noise added
noise_type=normal
sampling_method=normal
var_noise=0.1 # std for noise added 
# 0.1 for smt; 0.5 for sm
num_noise=100 # number for noise added

## other options
# -just_eval: no traing
# -use_true_type: Use ground truth event type for sampling

add_noise=denoise
parametrize=intensity
model=smurf_thp

## model saving
save_path=./checkpoints/mimic/

cd ../..


save_name=mimic_${model}_${parametrize}_${decoder}_${add_noise}_${noise_type}_${var_noise} # save model path
CUDA_VISIBLE_DEVICES=$device python main.py -loss_lambda $loss_lambda -normalize $normalize -sampling_method $sampling_method -decoder $decoder -parametrize $parametrize -add_noise $add_noise -var_noise $var_noise -num_noise $num_noise -n_save_steps $n_save_steps -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -langevin_step $langevin_step -n_steps $n_steps -n_samples $n_samples -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -save_path $save_path -save_name $save_name

load_path_name=./checkpoints/mimic/mimic_${model}_${parametrize}_${decoder}_${add_noise}_${noise_type}_${var_noise}.pth
save_result=./results/mimic/mimic_${model}_${parametrize}_${decoder}_${add_noise}_${sampling_method}_${var_noise}_samples
CUDA_VISIBLE_DEVICES=$device python main.py -loss_lambda $loss_lambda -just_eval -normalize $normalize -sampling_method $sampling_method -decoder $decoder -parametrize $parametrize -add_noise $add_noise -var_noise $var_noise -num_noise $num_noise -n_save_steps $n_save_steps -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -langevin_step $langevin_step -n_steps $n_steps -n_samples $n_samples -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -load_path_name $load_path_name -save_result $save_result

