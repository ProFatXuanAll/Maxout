import argparse
import json
import os

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
        '--ckpt_step',
        help='Checkpoint save interval.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--d_hid',
        help='Hidden layer dimension.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dataset',
        choices=src.dataset.dataset_map.keys(),
        help='Name of the dataset.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--exp_name',
        help='Name of the experiment.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--k',
        help='Number of sub-layer in hidden layers.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--n_layer',
        help='Number of hidden layers.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--log_step',
        help='Logging interval.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--lr',
        help='Learning rate.',
        required=True,
        type=float
    )
    parser.add_argument(
        '--max_norm',
        help='Maxnorm regularity.',
        required=True,
        type=float
    )
    parser.add_argument(
        '--momentum',
        help='SGD momentum.',
        required=True,
        type=float
    )
    parser.add_argument(
        '--seed',
        help='Control random seed.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--total_step',
        help='Total training step.',
        required=True,
        type=int
    )

    return parser.parse_args()


def main():
    # Parse argument.
    args = parse_args()

    # Get experiment path and loggin path.
    exp_path = os.path.join(src.path.EXP_PATH, args.exp_name)
    log_path = os.path.join(src.path.LOG_PATH, args.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Save configuration.
    with open(os.path.join(exp_path, 'cfg.json'), 'w') as output_file:
        json.dump(args.__dict__, output_file)

    # Set random seed.
    src.util.set_seed(seed=args.seed)

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load dataset.
    dataset_module = src.dataset.dataset_map[args.dataset]
    dataset = dataset_module.get_data(train=True)

    # Get mini-batch loader.
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Get model.
    model = src.model.MaxoutNN(
        d_in=dataset_module.get_d_in(),
        d_hid=args.d_hid,
        d_out=dataset_module.get_d_out(),
        k=args.k,
        n_layer=args.n_layer
    )
    model.train()
    model = model.to(device)

    # Get optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum
    )

    # Get objective.
    criterion = torch.nn.CrossEntropyLoss()

    # Get logger.
    writer = torch.utils.tensorboard.SummaryWriter(log_path)

    # Total loss.
    total_loss = 0.0
    pre_total_loss = 0.0

    # Training step counter.
    step = 0
    while step <= args.total_step:
        epoch_loader = tqdm(
            data_loader,
            desc=f'loss: {pre_total_loss:.6f}'
        )
        for x, y in epoch_loader:
            x = x.reshape(-1, dataset_module.get_d_in()).to(device)
            y = y.to(device)

            # Forward pass.
            loss = criterion(model(x), y)

            # Record average loss.
            total_loss += loss.item() / args.log_step

            # Backward pass.
            loss.backward()

            # Gradient descend.
            optimizer.step()

            # Clean up gradient.
            optimizer.zero_grad()

            # Increment training step.
            step += 1

            if step > args.total_step:
                break

            # Ckeckpoint save step.
            if step % args.ckpt_step == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, f'model-{step}.pt')
                )

            # Performance logging step.
            if step % args.log_step == 0:
                # Log average loss on CLI.
                epoch_loader.set_description(f'loss: {total_loss:.6f}')

                # Log average loss on tensorboard.
                writer.add_scalar('loss', total_loss, step)

                # Clean up average loss.
                pre_total_loss = total_loss
                total_loss = 0.0

    # Close logger.
    writer.close()


if __name__ == '__main__':
    main()
