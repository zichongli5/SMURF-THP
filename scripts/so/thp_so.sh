device=1
data=../data/data_so/fold1/
batch=4
n_head=4
n_layers=4
d_model=512
d_rnn=64
d_inner=1024
d_k=512
d_v=512
optimizer=adam
scheduler=reduce
dropout=0.1
lr=1e-4
smooth=0.1
epoch=60
eval_epoch=61

normalize=None

## langevin sampling parameter
langevin_step=3e-3 # sampling step size; decrease for better sampling, but cost more step
n_steps=1000 # sampling step count; increase for better sampling, but cost more time
n_save_steps=100 # save step per n_save_steps
n_samples=100 # number of generated sample; increase for more sample, but cost more memory
sampling_method=normal

save_path=./checkpoints/so/
save_name=so_thp_model_best_${normalize}

eval_quantile=-1
eval_quantile_step=0.05

cd ../..
model=thp

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py -normalize $normalize -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -save_name $save_name -save_path $save_path

load_path_name=./checkpoints/so/so_thp_model_best_${normalize}.pth
save_result=./results/so/so_thp_sample


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py  -normalize $normalize -sampling_method $sampling_method -save_result $save_result -just_eval -n_steps $n_steps -n_save_steps $n_save_steps -langevin_step $langevin_step -n_samples $n_samples -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -eval_epoch $eval_epoch -optimizer $optimizer -scheduler $scheduler -model $model -eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step -load_path_name $load_path_name
