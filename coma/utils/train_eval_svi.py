import time

from tqdm import tqdm


def run_svi(svi, train_loader, test_loader, epochs, device):  # , scheduler, writer):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train_eval_svi(svi, train_loader, device, train=True)
        t_duration = time.time() - t
        test_loss = train_eval_svi(svi, test_loader, device, train=False)
        # scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }
        print(info)
        # writer.print_info(info)
        # writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train_eval_svi(svi, loader, device, train=True):
    total_loss = 0.
    for data in tqdm(loader):
        x = data.x.to(device)
        if train:
            total_loss += svi.step(x)
        else:
            total_loss += svi.evaluate_loss(x)

    size = len(loader.dataset)
    total_epoch_loss = total_loss / size
    return total_epoch_loss
