import argparse

import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard

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
        '--p_in',
        help='Input units dropout rate.',
        required=True,
        type=float
    )
    parser.add_argument(
        '--p_hid',
        help='Hidden units dropout rate.',
        required=True,
        type=float
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

if __name__ == '__main__':
    # Parse argument.
    args = parse_args()

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
    model = src.model.DropoutNN(
        d_in=dataset_module.get_d_in(),
        d_hid=args.d_hid,
        d_out=dataset_module.get_d_out(),
        p_in=args.p_in,
        p_hid=args.p_hid
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

    # Training step counter.
    step = 0
    while step <= args.total_step:
        for x, y in data_loader:
            x = x.reshape(-1, dataset_module.get_d_in()).to(device)
            y = y.to(device)

            # Forward pass.
            loss = criterion(model(x), y)

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
