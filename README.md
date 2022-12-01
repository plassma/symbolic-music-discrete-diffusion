# SCHmUBERT
### a Symbolic Creative Harmonic music Unmasking Bidirectional Encoder Representation Transformer



https://user-images.githubusercontent.com/5902684/205120516-1c77b026-fa5c-4a0e-be68-1d1345e716be.mp4



Code for 2022 ISMIR LBD

## Installation
I run my experiments in Python 3.10, with all dependencies managed by Conda.

```conda env create -f env.yml```

Note that for all experiments, a soundfont-file called 'soundfont.sf2' (not included) must be located in the root-directory of the project.

## Prepare Dataset

I use the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) to train the models.
For loading, preprocessing and extracting melodies and trios from the MIDI files, I adapted the [pipelines magenta implemented for their MusciVAE](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae).
To prepare the dataset run:

```python prepare_data.py --root_dir=/path/to/lmd_full --target data/lakh_trio.npy --mode trio --bars 64```

## Train

I use [visdom](https://github.com/fossasia/visdom) to log the training progress and periodically show samples.

To train the model, start visdom and run for example:

```python train.py --dataset data/lakh_trio.npy --bars 64 --batch_size 64 --tracks trio --model conv_transformer```

So far, I got the best results with the conv_transformer model with one 1DConvolutional layer with a width of 4.
Pay attention to the ```steps_per_eval``` param, which is set to 10000 per default.
The evaluation step is more computationally expensive than training for 10000 steps, which is why you might want to increase this value if you do not need that many evaluations.


## Evaluate

To evaluate the framewise self-similarity metric on the samples generated by a model, run:

```python evaluate.py --mode unconditional|infilling|self```

## Sample

For sampling, I ~implemented~ hacked a rudimentary GUI using [nicegui](https://github.com/zauberzeug/nicegui).

```python sample.py --load_step 140000 --bars 64 --tracks trio --model conv_transformer```

The GUI supports:
  * visualizing samples (melody=red, bass=blue, drums=black), y position indicated pitch height, special pitch values: 0: pause, 1: note off, 90: mask
  * adaption of sample steps (Slider in Upload Expansion area)
  * diffuse from left to right ('=>') or vice versa ('<=')
  * copy from left to right ('>') or vice versa, only mask values are overwritten
  * sampling unconditionally (select 'A' in the central toggle to diffuse **A**ll (batch of 8) instead of the **S**elected sample)
  * uploading midi or musicxml - pieces for conditioning
  * masking whole tracks LM = Left Melody, RD = Right Drums, ....
  * masking area selected with mouse (mask button at the bottom)
  * playing with cursor indicating exact position in left and right visualization

Note: This sampling-tool was coded in very little time, and mainly coded for the LBD-Video.
It certainly needs to be reworked, but works if handled carefully.

## Samples

The following samples were produced using the model weights provided in the next section.
melody and trio were hand-selected as the best of 8 generated samples, the third sample was conditioned on a monophonic (melody only) MIDI piece (Super-Mario theme) with a length of 72 seconds. The remaining 56 seconds of the melody track, and all of the bass and drum track were generated by SCHmUBERT. The Mario sample was not hand-selected, but the first produced sample was uploaded (conditioned generation works more reliable and consistent than unconditional).

https://user-images.githubusercontent.com/5902684/205109877-49158518-321c-4533-9f47-e4e02102ad60.mp4

https://user-images.githubusercontent.com/5902684/205113067-3e208b03-959d-419c-90f7-147ff4fc438a.mp4

https://user-images.githubusercontent.com/5902684/205115137-d0742f99-a13e-4377-b056-175f37cea23f.mp4

## Model Weights

Model weights for the Conv_Transformer model trained on the Lakh-MIDI Dataset can be obtained [here](https://drive.google.com/file/d/1o719d4SOBMdjb3Gv_t6gxXVipekPdYhd/view?usp=share_link).
Extract the 'logs' folder to the project root, and set the ```load_step``` accordingly (180000 for melody, 140000 for trio).

