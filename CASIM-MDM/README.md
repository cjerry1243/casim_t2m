# Composite Aware Semantic Injection for MDM

This directory contains the code for training, evaluation, and demo that generates motions with CASIM-MDM. Two datasets HumanML3D and KIT-ML are both supported. The pretrained models are also released.

## Preparation

Follow the [Getting Started](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#getting-started) Section from the official MDM repo to setup the working environment, download dependencies for text-to-motion, preprocess the HumanML3D and KIT-ML datasets, and download their pretrained models (optional, for comparison only).

Then install the remaining packages with your environment.
```bash
pip install transformers
pip install wordcloud
pip install jupyter
```

## Pretrained CASIM-MDM Models

1. Download the model [checkpoints](https://rutgersconnect-my.sharepoint.com/:f:/g/personal/cc1845_cs_rutgers_edu/EkKs_2bjoTVKh3rjdgOlPlsB_RdnqsAN7rqVWgaDs_epsw?e=BTFhEJ), which include combinations of model variants (encoder, decoder) and diffusion steps (50, 1000).

2. Put all model checkpoints under `save/` folder

```bash
mkdir save/
mv YOUR_CKPT_DIR/CASIM-MDM* save/
```

## Generate samples with CASIM-MDM

- Generate from a single prompt

```bash
python -m sample.generate --model_path save/CASIM-MDM-Enc-1000steps/model000200000.pt --text_prompt "the person walks forward and is picking up his toolbox with his left hand." --use_casim --arch trans_enc --diffusion_steps 1000
```

- Generate from the example text file:

```bash
python -m sample.generate --model_path save/CASIM-MDM-Enc-1000steps/model000200000.pt  --input_text ./assets/casim_example_text_prompts.txt --use_casim --arch trans_enc --diffusion_steps 1000
```

- Generate from test set prompts:

```bash
python -m sample.generate --model_path save/CASIM-MDM-Enc-1000steps/model000200000.pt --num_samples 10 --num_repetitions 3 --use_casim --arch trans_enc --diffusion_steps 1000
```

The generated motions and stick figure visualization will be saved under `save/`.

You may also want to specify the motion lengths using `--motion_length` in seconds (default is 6.0 seconds, and max is 9.8 seconds).

### Render SMPL mesh

The above will generate motions in the format of npy files, which are compatible and can be rendered with SMPL mesh in the original [MDM](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#render-smpl-mesh) repo.


## Interactive Demo

TBD

## Evaluation

To evaluate the trained CASIM models, follow the instructions below.

### HumanML3D Dataset

- CASIM-MDM decoder variant with 1000 diffusion steps: (This can take a couple of days)

```bash
python -m eval.eval_humanml --model_path save/CASIM-MDM-DEC-1000steps/model000200000.pt --dataset humanml  --use_casim --arch trans_dec --diffusion_steps 1000 --eval_mode no_sample_limit
```

- CASIM-MDM decoder variant with 50 diffusion steps:

```bash
python -m eval.eval_humanml --model_path save/CASIM-MDM-DEC-50steps/model000100000.pt --dataset humanml  --use_casim --arch trans_dec --diffusion_steps 50 --eval_mode no_sample_limit
```

- CASIM-MDM encoder variant with 1000 diffusion steps:

```bash
python -m eval.eval_humanml --model_path save/CASIM-MDM-ENC-1000steps/model000200000.pt --dataset humanml  --use_casim --arch trans_enc --diffusion_steps 1000 --eval_mode no_sample_limit
```

- CASIM-MDM encoder variant with 50 diffusion steps:

```bash
python -m eval.eval_humanml --model_path save/CASIM-MDM-ENC-50steps/model000300000.pt --dataset humanml  --use_casim --arch trans_enc --diffusion_steps 50 --eval_mode no_sample_limit
```

### KIT-ML Dataset

Simply use the same commands above with the change `--dataset kit`. For example, for CASIM-MDM decoder variant with 1000 diffusion steps:

```bash
python -m eval.eval_humanml --model_path save/CASIM-MDM-KIT-DEC-1000Steps --dataset kit --diffusion_steps 1000 --use_casim --arch trans_dec --eval_mode no_sample_limit
```


## Training 

You may want to train your own model from scratch or with a customed dataset. 
Please follow the instructions below to reproduce our training results.

### HumanML3D Dataset

- CASIM-MDM decoder variant with 1000 diffusion steps:

```bash
python -m train.train_mdm --save_dir save/My-CASIM-MDM-DEC-1000Steps --dataset humanml --diffusion_steps 1000 --batch_size 512 --eval_during_training --num_steps 300000 --use_casim --arch trans_dec --overwrite
```

- CASIM-MDM decoder variant with 50 diffusion steps:

```bash
python -m train.train_mdm --save_dir save/My-CASIM-MDM-DEC-50Steps --dataset humanml --diffusion_steps 50 --batch_size 512 --eval_during_training --num_steps 300000 --use_casim --arch trans_dec --overwrite
```

- CASIM-MDM encoder variant with 1000 diffusion steps:

```bash
python -m train.train_mdm --save_dir save/My-CASIM-MDM-ENC-1000Steps --dataset humanml --diffusion_steps 1000 --batch_size 512 --eval_during_training --num_steps 300000 --use_casim --arch trans_enc --overwrite
```

- CASIM-MDM encoder variant with 50 diffusion steps:

```bash
python -m train.train_mdm --save_dir save/My-CASIM-MDM-ENC-50Steps --dataset humanml --diffusion_steps 50 --batch_size 512 --eval_during_training --num_steps 300000 --use_casim --arch trans_enc --overwrite
```

Usually it takes 200000 training iterations to achieve the best performance for model with 1000 diffusion steps and 300000 iterations for model with 50 diffusion steps.

### KIT-ML Dataset

Simply use the commands above with the change `--dataset kit`. For example, for CASIM-MDM decoder variant with 1000 diffusion steps:

```bash
python -m train.train_mdm --save_dir save/My-CASIM-MDM-KIT-DEC-1000Steps --dataset kit --diffusion_steps 1000 --batch_size 512 --eval_during_training --num_steps 300000 --use_casim --arch trans_dec --overwrite
```

It should take 100000 training steps on KIT-ML dataset to converge.


## Misc

This repo supports all command line args used in the original MDM repo. We also provide additional input args, described as follows:

- BERT text encoder:

Use `--user_bert` to specify the pretrained BERT text encoder instead of CLIP.

- Final projection for clip token embeddings:

Use `--clip_final_proj` to inject the CLIP token embeddings from the final layer for CASIM-MDM.

- Attention distribution visualization and analysis for CASIM.

We setup a hook to register the output of the CASIM attention scores for the decoder variant. To obtain the attention scores given an input text and conduct the analysis, go to the [attention_analysis.ipynb](./attention_analysis.ipynb) notebook.


## BibTex Citation

If you find our work useful, please cite with the following bibtex:

```BibTeX
@article{chang2025casim,
  title={CASIM: Composite Aware Semantic Injection for Text to Motion Generation},
  author={Chang, Che-Jui and Liu, Qingze Tony and Zhou, Honglu and Pavlovic, Vladimir and Kapadia, Mubbasir},
  journal={arXiv preprint arXiv:2502.02063},
  year={2025}
}
```

