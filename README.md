# Federated GANs (FeGAN)
Implementing GANs in a Federated Learning setup on PyTorch.

To run the single-machine version: python3 lsgan.py || python3 dcgan.py 	[See the script for options]

lsgan works only with MNIST, Fashion-MNIST, and Cifar10 -- dcgan works with CelebA and ImageNet

To run FeGAN: ./run.sh n E C B s w model max_samples machines iid || ./run_dcgan.sh (with the same parameters)  [See dist-lsgan.py || dist-dcgan.sh for options]

To run MD-GAN:	./run_md_gan.sh n model machines 		[see md-gan.py for options]

Args

n	the total number of processes in the experiment (=num_workers + 1)

E	the number of iterations done locally on workers at each FL round

C	the fraction of workers chosen in each FL round for training

B	the batch size

s	flag to enable the balanced sampling technique

w	flag to enable the KL weighting technique

model	the dataset to be used. Currently available datasets (MNIST, Fashion-MNIST, Cifar10, CelebA, ImageNet). Note that we do not provide the data in this repo. Data of the first three datasets are downloaded automatically by our script in a directory named "data". If one would like to try CelebA or Imagenet, she has to download data manually and put in a directory "data". Please see ``datasets.py'' for more information.

max_samples		the maximum number of samples per class per worker. This input is passed to the non-iidness engine described in the paper.

machines	the filename with the hostnames of nodes contributing to the experiment. Hostnames should be separated by newlines and no port number is required.

iid		flag to enable iid distribution of data on workers. If disabled, the non-iid engine distribute the data on workers

# Examples:

1) python3 lsgan.py --model mnist --batch 50

2) ./run_dcgan.sh 60 30 0.033 50 1 1 imagenet 50 machines 0

3) ./run_md_gan.sh 60 fashion-mnist machines

where, machines is the filename with hostnames contributing to this experiment
