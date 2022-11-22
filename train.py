import copy
import copy
import time

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from hparams import get_sampler_hparams
from utils import *
from utils.sampler_utils import get_samples
from utils.train_utils import EMA, optim_warmup


def main(H, vis):
    midi_data = np.load(H.dataset_path, allow_pickle=True)
    midi_data = SubseqSampler(midi_data, H.NOTES)

    val_idx = int(len(midi_data) * H.validation_set_size)
    train_loader = torch.utils.data.DataLoader(midi_data[val_idx:], batch_size=H.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=32)
    val_loader = torch.utils.data.DataLoader(midi_data[:val_idx], batch_size=H.batch_size)

    log(f'Total train batches: {len(train_loader)}, eval: {len(val_loader)}')

    sampler = get_sampler(H).cuda()
    optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # initialise before loading so as not to overwrite loaded stats
    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    cons_var = np.array([]), np.array([]), np.array([]), np.array([])
    start_step = 0

    if H.load_step > 0:
        start_step = H.load_step + 1

        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
            except Exception:
                ema_sampler = copy.deepcopy(sampler)
        if H.load_optim:
            optim = load_model(
                optim, f'{H.sampler}_optim', H.load_step, H.load_dir)
            # only used when changing learning rates and reloading from checkpoint
            for param_group in optim.param_groups:
                param_group['lr'] = H.lr

        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if train_stats is not None:
            losses = train_stats["losses"]
            mean_losses = train_stats["mean_losses"]
            val_losses = train_stats["val_losses"]
            val_elbos = train_stats["val_elbos"]
            elbo = train_stats["elbo"]
            cons_var = train_stats["cons_var"]
            H.steps_per_log = train_stats["steps_per_log"]
            log_start_step = 0

            # initialise plots
            vis.line(
                mean_losses,
                list(range(log_start_step, start_step, H.steps_per_log)),
                win='loss',
                opts=dict(title='Loss')
            )
            vis.line(
                elbo,
                list(range(log_start_step, start_step, H.steps_per_log)),
                win='ELBO',
                opts=dict(title='ELBO')
            )
            vis.line(
                val_losses,
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Val_loss',
                opts=dict(title='Validation Loss')
            )
            vis.line(
                val_elbos,
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Val_elbo',
                opts=dict(title='Validation ELBO')
            )
            vis.line(
                cons_var[0],
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Pitch',
                name='consistency',
                opts=dict(title='Pitch')
            )
            vis.line(
                cons_var[1],
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Pitch',
                name='variance',
                update='append',
                opts=dict(title='Pitch')
            )
            vis.line(
                cons_var[2],
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Duration',
                name='consistency',
                opts=dict(title='Duration')
            )
            vis.line(
                cons_var[3],
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Duration',
                name='variance',
                update='append',
                opts=dict(title='Duration')
            )
        else:
            log('No stats file found for loaded model, displaying stats from load step only.')

    sampler = sampler.cuda()

    scaler = torch.cuda.amp.GradScaler()
    train_iterator = cycle(train_loader)

    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")

    for step in range(start_step, H.train_steps):
        sampler.train()
        if H.ema:
            ema_sampler.train()
        step_start_time = time.time()
        # lr warmup
        if H.warmup_iters:
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = next(train_iterator)
        x = x.cuda(non_blocking=True)

        if H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = sampler.train_iter(x)

            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = sampler.train_iter(x)

            if torch.isnan(stats['loss']).any():
                log(f'Skipping step {step} with NaN loss')
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

        sampler.eval()
        if H.ema:
            ema_sampler.eval()

        if step % H.steps_per_log == 0:
            step_time_taken = time.time() - step_start_time
            stats['step_time'] = step_time_taken
            mean_loss = np.mean(losses)
            stats['mean_loss'] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])

            vis.line(
                np.array([mean_loss]),
                np.array([step]),
                win='loss',
                update=('append' if step > 0 else 'replace'),
                opts=dict(title='Loss')
            )
            log_stats(step, stats)

            if H.sampler == 'absorbing':
                elbo = np.append(elbo, stats['vb_loss'].item())
                vis.bar(
                    sampler.loss_history,
                    list(range(sampler.loss_history.size(0))),
                    win='loss_bar',
                    opts=dict(title='loss_bar')
                )
                vis.line(
                    np.array([stats['vb_loss'].item()]),
                    np.array([step]),
                    win='ELBO',
                    update=('append' if step > 0 else 'replace'),
                    opts=dict(title='ELBO')
                )

        if H.ema and step % H.steps_per_update_ema == 0 and step:
            ema.update_model_average(ema_sampler, sampler)

        if step % H.steps_per_sample == 0 and step:
            log(f"Sampling step {step}")
            samples = get_samples(ema_sampler if H.ema else sampler, H.sample_steps, b=H.show_samples)
            save_samples(samples, step, H.log_dir)
            vis_samples(vis, samples, step)

        if H.steps_per_eval and step % H.steps_per_eval == 0 and step:
            min_step = H.steps_per_eval
            [[c_p, c_d], [v_p, v_d]] = evaluate(H, ema_sampler if H.ema else sampler)
            cons_var = tuple(np.append(x, y) for x, y in zip(cons_var, [c_p, v_p, c_d, v_d]))
            vis.line(
                np.array([c_p]),
                np.array([step]),
                win='Pitch',
                update=('append' if step > min_step else 'replace'),
                name='consistency',
                opts=dict(title='Pitch')
            )
            vis.line(
                np.array([v_p]),
                np.array([step]),
                win='Pitch',
                update=('append' if step > min_step else 'replace'),
                name='variance',
                opts=dict(title='Pitch')
            )
            vis.line(
                np.array([c_d]),
                np.array([step]),
                win='Duration',
                update=('append' if step > min_step else 'replace'),
                name='consistency',
                opts=dict(title='Duration')
            )
            vis.line(
                np.array([v_d]),
                np.array([step]),
                win='Duration',
                update=('append' if step > min_step else 'replace'),
                name='variance',
                opts=dict(title='Duration')
            )

            # calculate validation loss
            valid_loss, valid_elbo, num_samples = 0.0, 0.0, 0
            log(f"Evaluating step {step}")

            for x in tqdm(val_loader):
                with torch.no_grad():
                    stats = sampler.train_iter(x.cuda())
                    valid_loss += stats['loss'].item()
                    if H.sampler == 'absorbing':
                        valid_elbo += stats['vb_loss'].item()
                    num_samples += x.size(0)
            valid_loss = valid_loss / num_samples
            if H.sampler == 'absorbing':
                valid_elbo = valid_elbo / num_samples

            val_losses = np.append(val_losses, valid_loss)
            val_elbos = np.append(val_elbos, valid_elbo)
            vis.line(
                np.array([valid_loss]),
                np.array([step]),
                win='Val_loss',
                update=('append' if step > min_step else 'replace'),
                opts=dict(title='Validation Loss')
            )
            if H.sampler == 'absorbing':
                vis.line(
                    np.array([valid_elbo]),
                    np.array([step]),
                    win='Val_elbo',
                    update=('append' if step > min_step else 'replace'),
                    opts=dict(title='Validation ELBO')
                )

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            save_model(optim, f'{H.sampler}_optim', step, H.log_dir)

            if H.ema:
                save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

            train_stats = {
                'losses': losses,
                'mean_losses': mean_losses,
                'val_losses': val_losses,
                'elbo': elbo,
                'val_elbos': val_elbos,
                'cons_var': cons_var,
                'steps_per_log': H.steps_per_log,
                'steps_per_eval': H.steps_per_eval,
            }
            save_stats(H, train_stats, step)


if __name__ == '__main__':
    H = get_sampler_hparams('train')
    vis = set_up_visdom(H)

    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, vis)
