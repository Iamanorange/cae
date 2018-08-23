# lossy image compression with compressive autoencoders

# results
![1](https://i.imgur.com/GWDbay4.png)
![2](https://i.imgur.com/KNi7fkh.jpg)
![3](https://i.imgur.com/LDSoBKb.jpg)
![4](https://i.imgur.com/cBJbLKg.jpg)
![5](https://i.imgur.com/ARbPB86.jpg)

# Environments
- GTX 1080 Ti (with 11GB graphic memory)
- Ubuntu 16.04
- Python 3.5
- Cuda 9.0
- Pytorch 0.4.1

# Preparing dataset
python3 resize.py --dataset_path ./dataset/Kodak

# Training
python3 train.py --exp_name Kodak --dataset ./dataset/Kodak

# Testing
