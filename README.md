# Maxout

Maxout Networks implemented with PyTorch.

Currently I only implemented MNIST experiment.

## Install

```sh
# Clone the repo.
git clone https://github.com/ProFatXuanAll/Maxout.git

# Install dependencies.
pipenv install
```

## Train

### Baseline

```sh
# Training script.
python run_train_base.py \
--batch_size 32 \
--ckpt_step 5000 \
--d_hid 1024 \
--dataset mnist \
--exp_name baseline \
--log_step 2500 \
--lr 1e-4 \
--max_norm 2 \
--momentum 0.95 \
--n_layer 1 \
--seed 42 \
--total_step 100000
```

### Dropout

```sh
# Training script.
python run_train_dropout.py \
--batch_size 32 \
--ckpt_step 5000 \
--d_hid 1024 \
--dataset mnist \
--exp_name dropout \
--log_step 2500 \
--lr 1e-4 \
--max_norm 2 \
--momentum 0.95 \
--n_layer 1 \
--p_in 0.2 \
--p_hid 0.5 \
--seed 42 \
--total_step 100000
```

### Maxout

```sh
# Training script.
python run_train_maxout.py \
--batch_size 32 \
--ckpt_step 5000 \
--d_hid 1024 \
--dataset mnist \
--exp_name maxout \
--k 4 \
--log_step 2500 \
--lr 1e-4 \
--max_norm 2 \
--momentum 0.95 \
--n_layer 1 \
--seed 42 \
--total_step 100000
```

### Maxout with Dropout

```sh
# Training script.
python run_train_maxout_with_dropout.py \
--batch_size 32 \
--ckpt_step 5000 \
--d_hid 1024 \
--dataset mnist \
--exp_name maxout_wdp \
--k 4 \
--log_step 2500 \
--lr 1e-4 \
--max_norm 2 \
--momentum 0.95 \
--n_layer 1 \
--p_in 0.2 \
--p_hid 0.5 \
--seed 42 \
--total_step 100000
```

## Evaluate

### Baseline

```sh
# Evaluation script on training set.
python run_eval_base.py \
--batch_size 128 \
--exp_name baseline \
--is_train

# Evaluation script on testing set.
python run_eval_base.py \
--batch_size 128 \
--exp_name baseline
```

### Dropout

```sh
# Evaluation script on training set.
python run_eval_dropout.py \
--batch_size 128 \
--exp_name dropout \
--is_train

# Evaluation script on testing set.
python run_eval_dropout.py \
--batch_size 128 \
--exp_name dropout
```

### Maxout

```sh
# Evaluation script on training set.
python run_eval_maxout.py \
--batch_size 128 \
--exp_name maxout \
--is_train

# Evaluation script on testing set.
python run_eval_maxout.py \
--batch_size 128 \
--exp_name maxout
```

### Maxout with Dropout

```sh
# Evaluation script on training set.
python run_eval_maxout_with_dropout.py \
--batch_size 128 \
--exp_name maxout_wdp \
--is_train

# Evaluation script on testing set.
python run_eval_maxout_with_dropout.py \
--batch_size 128 \
--exp_name maxout_wdp
```

## Experiment Results

### MNIST

#### Shared Configuration

|batch_size|ckpt_step|log_step|lr|max_norm|momentum|seed|total_step|
|-|-|-|-|-|-|-|-|
|32|5000|2500|1e-4|2|0.95|42|100000|

#### Results & Configuration

|model|train-acc|test-acc|d_hid|k|n_layer|p_in|p_hid|
|-|-|-|-|-|-|-|-|
|baseline|0.9547|0.9537|1024|N/A|1|N/A|N/A|
|baseline|0.9529|0.9536|1200|N/A|1|N/A|N/A|
|baseline|0.9604|0.9571|4096|N/A|1|N/A|N/A|
|baseline|0.9707|0.9656|1024|N/A|2|N/A|N/A|
|baseline|0.9710|0.9661|1200|N/A|2|N/A|N/A|
|baseline|0.9757|0.9689|4096|N/A|2|N/A|N/A|
|dropout|0.9571|0.9573|1024|N/A|1|0.2|0.5|
|dropout|0.9578|0.9566|1200|N/A|1|0.2|0.5|
|dropout|0.9604|0.9590|4096|N/A|1|0.2|0.5|
|dropout|0.9690|0.9661|1024|N/A|2|0.2|0.5|
|dropout|0.9690|0.9658|1200|N/A|2|0.2|0.5|
|dropout|0.9726|0.9696|4096|N/A|2|0.2|0.5|
|maxout|0.9705|0.9656|256|4|1|N/A|N/A|
|maxout|0.9732|0.9670|1024|4|1|N/A|N/A|
|maxout|0.9740|0.9685|1200|4|1|N/A|N/A|
|maxout|0.9829|0.9749|256|4|2|N/A|N/A|
|maxout|0.9861|0.9763|1024|4|2|N/A|N/A|
|maxout|0.9871|0.9761|1200|4|2|N/A|N/A|
|maxout+dropout|0.|0.|1024|4|1|N/A|N/A|
