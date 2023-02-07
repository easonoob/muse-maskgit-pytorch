<img src="./muse.png" width="450px"></img>

## Implementation - Muse - Python Library made by Lucidrains

Implementation of [Muse Python library made by Lucidrains](https://github.com/lucidrains/muse-maskgit-pytorch)

## Usage

First, install all the necessary libraries:\
```pip install -r requirements.txt```

To run the program:\
```python muse_train.py --image_size 512 --data_folder path/to/your/dataset/```

Other optional arguments:
```
  -h, --help            show this help message and exit
  --resume_from RESUME_FROM
                        Path to the vae model. eg. 'results/vae.steps.pt'
  --data_folder DATA_FOLDER
                        Dataset folder where your input images for training are.
  --num_train_steps NUM_TRAIN_STEPS
                        Total number of steps to train for. eg. 50000.
  --dim DIM             Model dimension.
  --batch_size BATCH_SIZE
                        Batch Size.
  --lr LR               Learning Rate.
  --grad_accum_every GRAD_ACCUM_EVERY
                        Gradient Accumulation.
  --save_results_every SAVE_RESULTS_EVERY
                        Save results every this number of steps.
  --save_model_every SAVE_MODEL_EVERY
                        Save the model every this number of steps.
  --vq_codebook_size VQ_CODEBOOK_SIZE
                        Image Size.
  --image_size IMAGE_SIZE
                        Image size. You may want to start with small images, and then curriculum learn to larger ones, but because
                        the vae is all convolution, it should generalize to 512 (as in paper) without training on it
  --epochs EPOCHS       Number of epochs to train for.
  --base_texts BASE_TEXTS
                        List of Prompts to use.
  --base_resume_from BASE_RESUME_FROM
                        Path to the vae model. eg. 'results/vae.steps.pt'
  --base_num_tokens BASE_NUM_TOKENS
                        must be same as vq_codebook_size.
  --base_seq_len BASE_SEQ_LEN
                        must be equivalent to fmap_size ** 2 in vae.
  --base_dim BASE_DIM   Model dimension.
  --base_depth BASE_DEPTH
                        Depth.
  --base_dim_head BASE_DIM_HEAD
                        Attention head dimension.
  --base_heads BASE_HEADS
                        Attention heads.
  --base_ff_mult BASE_FF_MULT
                        Feedforward expansion factor
  --base_t5_name BASE_T5_NAME
                        Name of your T5 model.
  --base_vq_codebook_size BASE_VQ_CODEBOOK_SIZE
  --base_image_size BASE_IMAGE_SIZE
  --base_cond_drop_prob BASE_COND_DROP_PROB
                        Conditional dropout, for Classifier Free Guidance
  --base_cond_scale BASE_COND_SCALE
                        Conditional for Classifier Free Guidance
  --base_timesteps BASE_TIMESTEPS
                        Time Steps to use for the generation.
  --superres_texts SUPERRES_TEXTS
                        List of Prompts to use.
  --superres_resume_from SUPERRES_RESUME_FROM
                        Path to the vae model. eg. 'results/vae.steps.pt'
  --superres_num_tokens SUPERRES_NUM_TOKENS
                        must be same as vq_codebook_size.
  --superres_seq_len SUPERRES_SEQ_LEN
                        must be equivalent to fmap_size ** 2 in vae.
  --superres_dim SUPERRES_DIM
                        Model dimension.
  --superres_depth SUPERRES_DEPTH
                        Depth.
  --superres_dim_head SUPERRES_DIM_HEAD
                        Attention head dimension.
  --superres_heads SUPERRES_HEADS
                        Attention heads.
  --superres_ff_mult SUPERRES_FF_MULT
                        Feedforward expansion factor
  --superres_t5_name SUPERRES_T5_NAME
                        name of your T5
  --superres_vq_codebook_size SUPERRES_VQ_CODEBOOK_SIZE
  --superres_image_size SUPERRES_IMAGE_SIZE
  --prompt PROMPT       List of Prompts to use for the generation.
  --base_model_path BASE_MODEL_PATH
                        Path to the base vae model. eg. 'results/vae.steps.base.pt'
  --superres_maskgit SUPERRES_MASKGIT
                        Path to the superres vae model. eg. 'results/vae.steps.superres.pt'
```

# Original README.md:

## Muse - Pytorch

Implementation of [Muse Python library made by Lucidrains](https://github.com/lucidrains/muse-maskgit-pytorch)

## Usage

First, install all the necessary libraries:\
```pip install -r requirements.txt```

To run the program:\
```python muse_train.py --image_size 512 --data_folder path/to/your/dataset/```

Other optional arguments:
```
  -h, --help            show this help message and exit
  --resume_from RESUME_FROM
                        Path to the vae model. eg. 'results/vae.steps.pt'
  --data_folder DATA_FOLDER
                        Dataset folder where your input images for training are.
  --num_train_steps NUM_TRAIN_STEPS
                        Total number of steps to train for. eg. 50000.
  --dim DIM             Model dimension.
  --batch_size BATCH_SIZE
                        Batch Size.
  --lr LR               Learning Rate.
  --grad_accum_every GRAD_ACCUM_EVERY
                        Gradient Accumulation.
  --save_results_every SAVE_RESULTS_EVERY
                        Save results every this number of steps.
  --save_model_every SAVE_MODEL_EVERY
                        Save the model every this number of steps.
  --vq_codebook_size VQ_CODEBOOK_SIZE
                        Image Size.
  --image_size IMAGE_SIZE
                        Image size. You may want to start with small images, and then curriculum learn to larger ones, but because
                        the vae is all convolution, it should generalize to 512 (as in paper) without training on it
  --epochs EPOCHS       Number of epochs to train for.
  --base_texts BASE_TEXTS
                        List of Prompts to use.
  --base_resume_from BASE_RESUME_FROM
                        Path to the vae model. eg. 'results/vae.steps.pt'
  --base_num_tokens BASE_NUM_TOKENS
                        must be same as vq_codebook_size.
  --base_seq_len BASE_SEQ_LEN
                        must be equivalent to fmap_size ** 2 in vae.
  --base_dim BASE_DIM   Model dimension.
  --base_depth BASE_DEPTH
                        Depth.
  --base_dim_head BASE_DIM_HEAD
                        Attention head dimension.
  --base_heads BASE_HEADS
                        Attention heads.
  --base_ff_mult BASE_FF_MULT
                        Feedforward expansion factor
  --base_t5_name BASE_T5_NAME
                        Name of your T5 model.
  --base_vq_codebook_size BASE_VQ_CODEBOOK_SIZE
  --base_image_size BASE_IMAGE_SIZE
  --base_cond_drop_prob BASE_COND_DROP_PROB
                        Conditional dropout, for Classifier Free Guidance
  --base_cond_scale BASE_COND_SCALE
                        Conditional for Classifier Free Guidance
  --base_timesteps BASE_TIMESTEPS
                        Time Steps to use for the generation.
  --superres_texts SUPERRES_TEXTS
                        List of Prompts to use.
  --superres_resume_from SUPERRES_RESUME_FROM
                        Path to the vae model. eg. 'results/vae.steps.pt'
  --superres_num_tokens SUPERRES_NUM_TOKENS
                        must be same as vq_codebook_size.
  --superres_seq_len SUPERRES_SEQ_LEN
                        must be equivalent to fmap_size ** 2 in vae.
  --superres_dim SUPERRES_DIM
                        Model dimension.
  --superres_depth SUPERRES_DEPTH
                        Depth.
  --superres_dim_head SUPERRES_DIM_HEAD
                        Attention head dimension.
  --superres_heads SUPERRES_HEADS
                        Attention heads.
  --superres_ff_mult SUPERRES_FF_MULT
                        Feedforward expansion factor
  --superres_t5_name SUPERRES_T5_NAME
                        name of your T5
  --superres_vq_codebook_size SUPERRES_VQ_CODEBOOK_SIZE
  --superres_image_size SUPERRES_IMAGE_SIZE
  --prompt PROMPT       List of Prompts to use for the generation.
  --base_model_path BASE_MODEL_PATH
                        Path to the base vae model. eg. 'results/vae.steps.base.pt'
  --superres_maskgit SUPERRES_MASKGIT
                        Path to the superres vae model. eg. 'results/vae.steps.superres.pt'
                        ```
