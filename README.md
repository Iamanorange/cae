# Lossy Image Compression with Compressive Autoencoders


See [wiki](https://github.com/alexandru-dinu/cae/wiki) for more details and further results.


## Results

`cae_32x32x32_zero_pad_bin` model, after roughly 5.8 millions of optimization steps;
left: original, right: reconstructed.

![](https://i.imgur.com/GWDbay4.png)
![](https://i.imgur.com/KNi7fkh.jpg)
![](https://i.imgur.com/LDSoBKb.jpg)
![](https://i.imgur.com/cBJbLKg.jpg)
![](https://i.imgur.com/ARbPB86.jpg)

## Resources

A smaller dataset (2,286 frames) that I have used in the 
[Further results](https://github.com/alexandru-dinu/cae/wiki/Further-results) 
page of the wiki can be downloaded [here](https://mega.nz/#!XU8EDCII!ZsCVLwobtZ8cWAOqRWr1qLAnn_NgVUvFhACs51EZiX8).

A bigger dataset can be constructed by downloading frames using the scripts provided [here](https://github.com/gsssrao/youtube-8m-videos-frames).
For the above results, I have randomly selected and downloaded 121,827 frames.


## Environments

- GTX 1080 Ti (with 11GB graphic memory)
- Ubuntu 16.04
- Python 3.5
- Cuda 9.0
- Pytorch 0.4.1


## Training

python3 train.py --exp_name Kodak --shuffle --dataset ./dataset/Kodak


## Testing

python3 test.py --chkpt ./checkpoints/Kodak/model_final.state --shuffle --dataset_path ./dataset/Kodak --out_dir ./Kodak


## References

[1] https://arxiv.org/abs/1703.00395

[2] http://arxiv.org/abs/1511.06085
