import os
import torch
import visdom
import numpy as np
import logging
from preprocessing.data import TrioConverter, OneHotMelodyConverter
from note_seq import fluidsynth, note_sequence_to_midi_file


def log(output):
    logging.info(output)
    print(output)


def config_log(log_dir, filename="log.txt"):
    log_dir = "logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def start_training_log(hparams):
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")


def save_model(model, model_save_name, step, log_dir):
    log_dir = "logs/" + log_dir + "/saved_models"
    os.makedirs(log_dir, exist_ok=True)
    model_name = f"{model_save_name}_{step}.th"
    print(f"Saving {model_save_name} to {model_save_name}_{str(step)}.th")
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir, strict=False):
    print(f"Loading {model_load_name}_{str(step)}.th")
    log_dir = "logs/" + log_dir + "/saved_models"
    try:
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th")),
            strict=strict,
        )
    except TypeError:  # for some reason optimisers don't liek the strict keyword
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th")),
        )

    return model


def save_samples(np_samples, step, log_dir):
    log_dir = "logs/" + log_dir + "/samples"
    os.makedirs(log_dir, exist_ok=True)
    np.save(log_dir + f'/samples_{step}.npz', np_samples, allow_pickle=True)


def save_stats(H, stats, step):
    save_dir = f"logs/{H.log_dir}/saved_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"logs/{H.log_dir}/saved_stats/stats_{step}"
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)


def save_noteseqs(ns, prefix='pre_adv'):
    for i, n in enumerate(ns):
        note_sequence_to_midi_file(n, prefix + f'_{i}.mid')


def samples_2_noteseq(np_samples):
    if np_samples.shape[2] == 3:
        converter = TrioConverter(16)#todo: Hparams, async
    else:
        converter = OneHotMelodyConverter()
        np_samples = np_samples[:, :, 0]
    return converter.from_tensors(np_samples)


def sample_audio(samples):
    samples = samples.copy()
    if len(samples.shape) < 3:
        samples = np.expand_dims(samples, 0)
    samples[samples == 90] = 0
    samples[samples == 512] = 0
    samples = samples_2_noteseq(samples)
    return [fluidsynth(s, 44100., 'soundfont.sf2') for s in samples]


def vis_samples(vis, samples, step):
    audios = sample_audio(samples)

    for i, audio in enumerate(audios):
        t = f'sample_{i}'
        try:
            vis.audio(audio, win=t, env=f'samples_{step}', opts=dict(title=t))
        except Exception as e:
            log(f'could not vis audio sample {i} for step {step}')
            log(e)


def load_stats(H, step):
    load_path = f"logs/{H.load_dir}/saved_stats/stats_{step}"
    stats = torch.load(load_path)
    return stats


def log_stats(step, stats):
    log_str = f"Step: {step}  "
    for stat in stats:
        if "latent_ids" not in stat:
            try:
                log_str += f"{stat}: {stats[stat]:.4f}  "
            except TypeError:
                log_str += f"{stat}: {stats[stat].mean().item():.4f}  "

    log(log_str)


def set_up_visdom(H):
    server = H.visdom_server
    try:
        if server:
            vis = visdom.Visdom(server=server, port=H.visdom_port)
        else:
            vis = visdom.Visdom(port=H.visdom_port)
        return vis

    except Exception:
        log_str = "Failed to set up visdom server - aborting"
        print(log_str, level="error")
        raise RuntimeError(log_str)