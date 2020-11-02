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

### Dropout

```sh
# Training script.
python run_train_dropout.py \
--batch_size 32 \
--ckpt_step 5000 \
--d_hid 1024 \
--dataset mnist \
--exp_name dropout \
--p_in 0.2 \
--p_hid 0.5 \
--log_step 2500 \
--lr 1e-4 \
--max_norm 2 \
--momentum 0.95 \
--seed 42 \
--total_step 100000
```

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
