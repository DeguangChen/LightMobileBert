# LightMobileBert
LightMobileBert is Secondary Lightweight Model based on MobileBert.


## TF-to-pytorch Pre-trained Checkpoints
Convert the checkpoints in tensorflow to the checkpoints in pytorch.<br>

* MobileBert Optimized Uncased English: [uncased_L-24_H-128_B-512_A-4_F-4_OPT](https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT.tar.gz)<br>
* run convert_tf_checkpoint_to_pytorch.py


## Finetune with LightMobileBert
* run run_glue.py (Because of the mobilebert checkpoints only have TF type).

## requirements
torch = 1.1.0 to 1.10.0<br>
transformers = 3.4.0<br>
scikit-learn = 1.1.1<br>



