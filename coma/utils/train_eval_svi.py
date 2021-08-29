from collections import defaultdict

import torch
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm


def run_svi(svi, model, train_loader, test_loader, epochs, scheduler, device, output_particles, writer):
    train_losses, test_losses = [], []

    train_recon_no = min(train_loader.batch_size, 10)
    test_recon_no = min(test_loader.batch_size, 10)

    for epoch in range(1, epochs + 1):
        train_metrics, train_recon_sample = train_eval_svi(
            svi, model, train_loader, device, output_particles, train=True
        )
        scheduler.step()
        writer.write_scalars(epoch, train=True, **train_metrics)
        writer.write_meshes(epoch, train_recon_sample[:train_recon_no], train=True)
        writer.save_model_checkpoint(model, epoch)

        test_metrics, test_recon_sample = train_eval_svi(
            svi, model, test_loader, device, output_particles, train=False
        )
        writer.write_scalars(epoch, train=False, **test_metrics)
        writer.write_meshes(epoch, test_recon_sample[:test_recon_no], train=True)

def train_eval_svi(svi, model, loader, device, output_particles, train=True):
    total_metrics = defaultdict(float)
    for data in tqdm(loader):
        x = data.x.to(device)
        if train:
            _step = svi.step(x)
        l = svi.evaluate_loss(x)
        total_metrics['loss'] += l 
        metrics = {
            **get_svi_metrics(svi),
            **get_recon_metrics(model, x, output_particles)
        }
        for k, v in metrics.items():
            total_metrics[k] += v

    size = len(loader.dataset)
    total_metrics = {k: v / size for k, v in total_metrics.items()}

    recon = get_recon(model, x)

    return total_metrics, recon

def get_svi_metrics(svi):
    metrics = {}

    model = svi.loss_class.trace_storage['model']
    guide = svi.loss_class.trace_storage['guide']

    # log likelihood of meshes
    metrics['log p(x)'] = model.nodes['x']['log_prob'].mean()
    # KL divergence
    metrics['log p(z)'] = model.nodes['z']['log_prob'].mean()
    metrics['log q(z)'] = guide.nodes['z']['log_prob'].mean()
    metrics['kl'] = metrics['log p(z)'] - metrics['log q(z)']

    return metrics

def get_recon_metrics(model, x, n_particles=1):
    recon = model.generate(x, n_particles)
    recon = recon.detach()

    mse = torch.nn.functional.mse_loss(recon, x)
    mae = torch.nn.functional.l1_loss(recon, x)
    chamfer = chamfer_distance(recon.float(), x.float())[0]

    return {
        'mse': mse,
        'mae': mae,
        'chamfer': chamfer,
    }

def get_recon(model, x, n_particles=1):
    return model.generate(x, n_particles).detach()
