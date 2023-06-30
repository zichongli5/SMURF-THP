## model parameter
device=2 # GPU device number
data=../data/data_bookorder/fold1/ # datasets directory
batch=1 # batch size
decoder=encode
n_head=6 # number of heads of transformer encoder
n_layers=6 # number of layers of transformer encoder
d_model=128 # dimension of the transformer model
d_rnn=64 # dimension of RNN layer (not used now)
d_inner=2048 # dimension of the hidden layer in transformer model(Feed Forward Network)
d_k=64 
d_v=64
optimizer=adam # optimizer
scheduler=cosLR # type of lr scheduler; 'steplr' for StepLR, 'reduce' for ReduceLROnPlateau
dropout=0.1
lr=1e-4
smooth=0.1 # smoothing of loss used for type prediction (0.1 is used in THP)
epoch=50
eval_epoch=71 # number of epoch we want to start to evaluate
normalize=log
loss_lambda=5

## langevin sampling parameter
langevin_step=1e-3 # sampling step size; decrease for better sampling, but cost more step
n_steps=3000 # sampling step count; increase for better sampling, but cost more time
n_save_steps=100 # save step per n_save_steps
n_samples=100 # number of generated sample; increase for more sample, but cost more memory

## quantile of calibration score
eval_quantile=-1 # -1 stand for np.arange(eval_quantile_step, 1, eval_quantile_step); could also be a float in (0,1)
eval_quantile_step=0.05 # used when eval_quantile=-1

## parameter for noise added
noise_type=normal
sampling_method=normal
var_noise=0.01 # std for noise added # 0.5 for type; 0.05 for time
num_noise=100 # number for noise added

## other options
# -just_eval: no traing
# -use_true_type: Use ground truth event type for sampling


## model saving
save_path=./checkpoints/financial/

cd ../..

add_noise=denoise
parametrize=intensity
model=smurf_thp

save_name=financial_${model}_${parametrize}_${decoder}_${add_noise}_${noise_type}_${var_noise}_${normalize} # save model path
CUDA_VISIBLE_DEVICES=$device python main.py -loss_lambda $loss_lambda -normalize $normalize -sampling_method $sampling_method -decoder $decoder -parametrize $parametrize -add_noise $add_noise -var_noise $var_noise -num_noise $num_noise -n_save_steps $n_save_steps -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -langevin_step $langevin_step -n_steps $n_steps -n_samples $n_samples -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -save_path $save_path -save_name $save_name

load_path_name=./checkpoints/financial/financial_${model}_${parametrize}_${decoder}_${add_noise}_${noise_type}_${var_noise}_${normalize}.pth
save_result=./results/financial/financial_${model}_${parametrize}_${decoder}_${add_noise}_${sampling_method}_${var_noise}_samples
CUDA_VISIBLE_DEVICES=$device python main.py -loss_lambda $loss_lambda -just_eval -normalize $normalize -sampling_method $sampling_method -decoder $decoder -parametrize $parametrize -add_noise $add_noise -var_noise $var_noise -num_noise $num_noise -n_save_steps $n_save_steps -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -langevin_step $langevin_step -n_steps $n_steps -n_samples $n_samples -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -load_path_name $load_path_name -save_result $save_result

