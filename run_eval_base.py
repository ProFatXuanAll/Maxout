import argparse
import json
import os
import re

import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard

from tqdm import tqdm

import src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='Batch size.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--exp_name',
        help='Name of the experiment.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--is_train',
        action='store_true',
        help='Name of the experiment.'
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    # Parse argument.
    args = parse_args()

    # Get experiment path and loggin path.
    exp_path = os.path.join(src.path.EXP_PATH, args.exp_name)
    log_path = os.path.join(src.path.LOG_PATH, args.exp_name)
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f'{exp_path} does not exist.')

    # Load configuration.
    with open(os.path.join(exp_path, 'cfg.json'), 'r') as input_file:
        cfg = argparse.Namespace(**json.load(input_file))

    # Set random seed.
    src.util.set_seed(seed=cfg.seed)

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load dataset.
    dataset_module = src.dataset.dataset_map[cfg.dataset]
    dataset = dataset_module.get_data(train=args.is_train)

    # Get mini-batch loader.
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Get all model checkpoint.
    pattern = re.compile(r'model-(\d+).pt')
    all_ckpts = filter(
        lambda ckpt: pattern.match(ckpt),
        os.listdir(exp_path)
    )
    all_ckpts = sorted(
        map(lambda ckpt: int(pattern.match(ckpt)[1]), all_ckpts)
    )

    # Get logger.
    writer = torch.utils.tensorboard.SummaryWriter(log_path)

    # Evaluate each checkpoint.
    for ckpt in tqdm(all_ckpts):
        # Get model.
        model = src.model.BaseNN(
            d_in=dataset_module.get_d_in(),
            d_hid=cfg.d_hid,
            d_out=dataset_module.get_d_out(),
            n_layer=cfg.n_layer
        )
        model.load_state_dict(torch.load(os.path.join(
            exp_path,
            f'model-{ckpt}.pt'
        )))
        model.eval()
        model = model.to(device)

        pred = []
        ans = []
        for x, y in data_loader:
            x = x.reshape(-1, dataset_module.get_d_in()).to(device)

            # Get prediction.
            pred.extend(model(x).argmax(dim=-1).to('cpu').tolist())

            # Get answer.
            ans.extend(y.tolist())

        # Log performance.
        writer.add_scalar(
            dataset_module.get_eval_name(train=args.is_train),
            dataset_module.get_eval(pred=pred, ans=ans),
            ckpt
        )

    # Close logger.
    writer.close()


if __name__ == '__main__':
    main()
