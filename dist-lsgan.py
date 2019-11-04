## This is an adapted version of lsgan.py file in implementations/lsgan/
## The goal is to write a distributed version of the implementation of lsgan
## The changed parts are preceeded with "#DIST" so that these parts can be copied/pasted in any of the provided gans implementation

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
#DIST
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
from datasets import DatasetManager
from fid_score import *
from inception import *
from time import sleep, time
import random
import sys
from scipy import stats
from queue import Queue

multiprocessing.set_sharing_strategy('file_system')
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

#DIST
""" Gradient averaging. """
def average_models(model, group=None, choose_r0=True, weights=None, elapsed_time=None):
    global fl_round
    global rat_per_class
    gp_size = len(all_groups_np[fl_round%len(all_groups)])
    if rank == 0 and opt.weight_avg and weights is not None:
        cur_gp = all_groups_np[fl_round%len(all_groups)]
        if opt.weight_scheme == 'exp':
            e_w = [np.exp(w.item()) for w in weights]               #Getting e^w for each w in weights (w here is the success rate of workers' generators)
        else:
            e_w = [w.item() for w in weights]

        e_w = np.array(e_w)
        if not choose_r0:
            e_w/= sum(e_w[1:])
        else:
            e_w/= sum(e_w)
        if opt.weight_scheme == 'dirac':
            e_w = [0 if w < 0.5 else w for w in e_w]		#The threshold here is 0.5
            #Reweighting after removing the harmful/useless updates (could work as a simulation to taking thee forgiving updates)
            if not choose_r0:
                e_w/= sum(e_w[1:])
            else:
                e_w/= sum(e_w)

#        print("Current group: ", cur_gp)
#        print("weights: ", e_w)

    for param in model.parameters():
        if rank == 0 and not choose_r0:				#If rank=0 is not in included in this round, put zeros instead
            param.data = torch.zeros(param.size()).cuda()
        if not opt.weight_avg or weights is None:
            dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
            param.data /= (gp_size if choose_r0 else gp_size - 1)
        else:
            gather_list = []
            if rank == 0:
                gather_list = [torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size()) for _ in range(gp_size)]
            dist.gather(param.data, gather_list, dst=0, group=group)
            if rank == 0:
                param.data = torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size())
                for w,t in zip(e_w,gather_list):
                    param.data+= t*w

#DIST
""" Model broadcast. """
def broadcast_model(model, group=None, elapsed_time=None):
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)

all_groups = []
all_groups_np = []
choose_r = []
fl_round = -1
def init_groups(size, cls_freq_wrk):
    """ 
	Initialization of all distributed groups for the whole training process. We do this in advance so as not to hurt the performance of training.
	The server initializes the group and send it to all workers so that everybody can agree on the working group at some round.
	Args
		size		The total number of machines in the current setup
		cls_freq_wrk	The frequency of samples of each class at each worker. This is used when the "sample" option is chosen. Otherwise, random sampling is applied and this parameter is not used. 
    """
    global all_groups
    global all_groups_np
    global choose_r
    all_groups = []
    all_groups_np = []
    choose_r = []
    done = False
    gp_size = int(opt.frac_workers*(size))
    #If opt.sample is set, use the smart sampling, i.e., based on frequency of samples of each class at each worker. Otherwise, use random sampling
    if opt.sample:
        #2D array that records if class i exists at worker j or not
        wrk_cls = [[False for i in range(10)] for j in range(size)]
        cls_q = [Queue(maxsize=size) for _ in range(10)]
        for i,cls_list in enumerate(cls_freq_wrk):
            wrk_cls[i] = [True if freq != 0 else False for freq in cls_list]
        for worker,class_list in enumerate(reversed(wrk_cls)):
            for cls,exist in enumerate(class_list):
                if exist:
                    cls_q[cls].put(size - worker-1)
	#This array counts the number of samples (per class) taken for training so far. The algorithm will try to make the numbers in this array as equal as possible
        taken_count = [0 for i in range(10)]
    while not done:
        if not opt.sample or rank != 0:			#It does not matter what other workers to.....only the server is required to create correct groups
            g = random.sample(range(0, size), gp_size)
        else:
            visited = [False for i in range(size)]	#makes sure that we take any worker only once in the group
            g = []
            for _ in range(gp_size):
                #Choose class (that is minimum represnted so far)...using "taken_count" array
                cls = np.where(taken_count == np.amin(taken_count))[0][0]
                assert cls >= 0 and cls <= len(taken_count)
                #Choose a worker to represnt that class...using wrk_cls and visited array
                done_q = False
                count = 0
                while not done_q:
                    wrkr = cls_q[cls].get()
                    assert wrk_cls[wrkr][cls]
                    if not visited[wrkr] and wrk_cls[wrkr][cls]:
                        #Update the state: taken_count and visited
                        g.append(wrkr)
                        taken_count += cls_freq_wrk[wrkr]
                        visited[wrkr] = True
                        done_q = True
                    cls_q[cls].put(wrkr)
                    count+=1
                    if count == size:	#Such an optimal assigm,emt does not exist
                        done_q = True
        choose_r0 = False
        if rank == 0:
            if 0 in g:
                choose_r0 = True
            else:
                choose_r0 = False
            choose_r.append(choose_r0)
        g.append(0)
        assert len(g) > 1
        if cuda:
            g = torch.cuda.FloatTensor(g)
        else:
            g = torch.FloatTensor(g)
#        print("Rank: {}, broadcast group number {}".format(rank, len(all_groups)))
        dist.broadcast(g,src=0)
        g = g.cpu().numpy().tolist()
        if g.count(0) > 1:                              #Make sure there is at most one occurance of "0" in the list of group members
            g.remove(0)
        try:
            group = dist.new_group(g)
        except Exception as e:
            done = True
        all_groups_np.append(np.sort(g))
        all_groups.append(group)
#        if rank == 0:
#            print("Group number {} is {}".format(len(all_groups), g))
        if len(all_groups) > 100:
            done = True
#DIST
def gather_lbl_count(lbl_count):
    """ 
	This function gathers all labels counts from all workers at the server.
	Args:
		lbl_count: array of frequency of samples of each class at the current worker
	returns:
		workers_classes: array of arrays of labels counts of each class at the server
    """
    gather_list = []
    if rank == 0:
        gather_list = [torch.zeros(len(lbl_count)).cuda() if cuda else torch.zeros(len(lbl_count)) for _ in range(size)]
    dist.gather(torch.cuda.FloatTensor(lbl_count) if cuda else torch.FloatTensor(lbl_count), gather_list, dst=0)
    res = [count_list.cpu().detach().numpy() for count_list in gather_list]
    return res

#DIST
rat_per_class=[]
def run(rank, size):
    global fl_round
    global rat_per_class
    # !!! Minimizes MSE instead of BCE
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
#DIST (fix the path of data)
    manager = DatasetManager(opt.model, opt.batch_size, opt.img_size, size-1, size, rank, opt.iid)
    train_set, _ = manager.get_train_set(opt.max_samples)

    lbl_count = [0 for _ in range(10)]
    for i, (imgs, lbls) in enumerate(train_set):
        for lbl in lbls:
            lbl_count[lbl.item()]+=1

    #This piece of info should be gathered at the server (to do informative decision about sampling)
    workers_classes = gather_lbl_count(lbl_count)
    if rank == 0:
        print(workers_classes)
    num_per_class = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]		#Aggregate number of classes is calculated manually here
    all_samples = sum(num_per_class)
    rat_per_class = [float(n/all_samples) for n in num_per_class]
    #Calculating entropy at this worker

    #Now, initializing all groups for the whole training process
#    gp_t = time()
    init_groups(size, workers_classes)
    print("Rank {} Done initializing {} groups".format(rank, len(all_groups)))
#    if opt.bench:
#        print("Time to init the groups: ", time() - gp_t)
    #Calculating entropy of each worker (on the server side) based on these frequencies....
    if rank == 0:
        entropies = [stats.entropy(np.array(freq_l)/sum(freq_l), rat_per_class) * (sum(freq_l)/ all_samples) for freq_l in workers_classes]	
        print("Entropies are: ", entropies)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    #For FID calculations
    if rank == 0:
        fic_model = InceptionV3()
        if cuda:
            fic_model = fic_model.cuda()
        test_set = manager.get_test_set()
        for i,t in enumerate(test_set):
            test_imgs = t[0].cuda()
            test_labels = t[1]
        grouped_test_imgages = [[] for i in range(10)]
        for i,img in enumerate(test_imgs):
            grouped_test_imgages[test_labels[i]].append(img)
        for i,arr in enumerate(grouped_test_imgages):
            grouped_test_imgages[i] = torch.stack(arr)

    # ----------
    #  Training
    # ----------
    #DIST
    elapsed_time = time()
    num_batches=0		#This variable acts as a global state variable to sync. between workers and the server
    done_round = True
    group = None
    #The following hack (4 lines) is written to run actually the number of runs that the user is aiming for....because of the skewness of data, the actual number of epochs that would run could be less than that the user is estimating...These few lines solve this issue
    est_len = 50000 // (size * opt.batch_size)		#Given a dataset of 50,000 imgaes, the estimated number of iterations to dataset is 50000/unm_workers
    act_len = len(train_set)
    if act_len < est_len:
        opt.n_epochs = int(opt.n_epochs * (est_len/act_len))
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_set):
            #DIST
            if done_round:		#This means that a new round should start....done by sampling a few of workers and give them the latest version of the model(s)
                #First step: Choose a group of nodes to do computations in this round....
                fl_round+=1
                g = all_groups_np[fl_round%len(all_groups)]
                group = all_groups[fl_round%len(all_groups)]
                choose_r0 = False
                if rank == 0:
                    choose_r0 = choose_r[fl_round%len(all_groups)]
#                broad_t = time()
                if rank in g:
                    broadcast_model(generator, group, elapsed_time)
                    broadcast_model(discriminator, group, elapsed_time)
                    done_round = False
                else:		#This node is not chosen in the current group....no work for this node in this round....just continue and wait for a new announcement from the server
                    done_round = True
                    num_batches=num_batches+opt.local_steps	#Advance the pointer for workers that will not work this round
                    continue
            num_batches+=1
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------
#            gen_t = time()
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
#            gd_t = time()
            d_gen = discriminator(gen_imgs)
            g_loss = adversarial_loss(d_gen, valid)

            g_loss.backward()
#            if opt.bench and rank == 0:
#                print("Time of bakward pass 1 for discriminator ", time() - gd_t)

            #DIST
#            g_avg_t = time()
            #Averaging step.......added because of distributed setup now!
            if num_batches%opt.local_steps == 0 and num_batches > 0:
                if opt.weight_avg:
                    #This is a weighting scheme using the entropies based on the frequency of samples of each class at each worker
                    cur_gp = all_groups_np[fl_round%len(all_groups)]
                    if rank  == 0:
                        weights = [entropies[int(wrk)] for wrk in cur_gp]
                    else:	#dummy else
                        weights = [1.0/len(cur_gp) for _ in cur_gp]
                    average_models(generator, group, choose_r0, weights, elapsed_time=elapsed_time)		#Experiments show that doing this is bad anyway!
                else:
                    average_models(generator, group, choose_r0, elapsed_time=elapsed_time)
                done_round = True
            if rank == 0 and not choose_r0:
                g_p = generator.parameters()
                for param in generator.parameters():
                    param.grad.data = torch.zeros(param.size()).cuda()
            
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------

#            disc_t = time()
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
    

            d_loss.backward()

            #DIST
            #Averaging step.......added because of distributed setup now!
#            d_avg_t = time()
            if num_batches%opt.local_steps == 0 and num_batches > 0:
                if opt.weight_avg:
                    average_models(discriminator, group, choose_r0, weights, elapsed_time=elapsed_time)
                else:
                    average_models(discriminator, group, choose_r0, elapsed_time=elapsed_time)
                done_round = True
            if rank == 0 and not choose_r0:
                for param in discriminator.parameters():
                    param.grad.data = torch.zeros(param.size()).cuda()
            optimizer_D.step()

            #Print stats and generate images only if this is the server
            batches_done = epoch * len(train_set) + i
            if rank == 0 and batches_done % opt.sample_interval == 0:
                print(
                    "Rank %d [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time %f"
                    % (rank, epoch, opt.n_epochs, i, len(train_set), d_loss.item(), g_loss.item(), time() - elapsed_time), 
                    end = ' ' if epoch != 0 else '\n'
                )
#                sys.stdout.flush()

                # Evaluation setp => output images and calculate FID
                if batches_done % opt.sample_interval == 0 and batches_done != 0:
#                    pathname = os.path.abspath(os.path.dirname(sys.argv[0]))
#                    save_image(gen_imgs.data[:25], pathname+"/images-dist-s{}-w{}/{}-{}.png".format(opt.sample, opt.weight_avg, rank,batches_done), nrow=5, normalize=True)
#                    print("=====Calculating FID for round {}======".format(fl_round))
                    fid_z = Variable(Tensor(np.random.normal(0, 1, (opt.fid_batch, opt.latent_dim))))
                    del gen_imgs
                    gen_imgs = generator(fid_z)
                    mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, fic_model)
                    mu_test, sigma_test = calculate_activation_statistics(test_imgs[:opt.fid_batch], fic_model)
                    fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
                    print("FL-round {} FID Score: {}".format(fl_round, fid))
                    sys.stdout.flush()
                    if False: #not opt.iid:
                        cur = 0
                        fids = [0 for i in range(10)]
                        for i,gp in enumerate(grouped_test_imgages):
                            mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs[cur:cur+len(gp)], fic_model)
                            cur += len(gp)
                            mu_test, sigma_test = calculate_activation_statistics(gp, fic_model)
                            fids[i] = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
                        print("avg: ", np.mean(fids), " max: ", np.max(fids), " min: ", np.min(fids))
#DIST
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

os.makedirs("images-dist", exist_ok=True)
    
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches (named B in FL notations)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="number of iterations to calculate the FID.")
#DIST
parser.add_argument("--model", type=str, default='mnist', help="model to train")
parser.add_argument("--local_steps", type=int, default=100, help="number of local steps to be executed in each worker before sending to the server (named E in FL notations).")
parser.add_argument("--frac_workers", type=float, default=0.1, help="fraction of workers that participate in each round computation (named C in FL notations).")
parser.add_argument("--fid_batch", type=int, default=4000, help="number of samples used to evaluate the progress of the GAN (using the FID score).")
parser.add_argument("--rank", type=int, default=-1, help="Rank of this node in the distributed setup.")
parser.add_argument("--size", type=int, default=-1, help="Total number of machines in this experiment.")
parser.add_argument("--iid", type=int, default=0, help="Determines whether data should be distributed in an iid fashion to all workers or not. Takes only 0 or 1 as a value.")
parser.add_argument("--weight_avg", type=int, default=0, help="If set, vanilla FL-GAN operates. Otherwise, the new weighted averaging with entropies scheme takes place.")
parser.add_argument("--sample", type=int, default=0, help="If set, smart sampling takes place. Otherwise, random sampling is used.")
parser.add_argument("--port", type=str, default='29500', help="Port number of the master....required for connections from everybody.")
parser.add_argument("--master", type=str, default='igrida-abacus9', help="The master hostname...should be known by everybody.")
#parser.add_argument("--bench", type=int, default=1, help="If set, time taken by each step is printed.")
parser.add_argument("--weight_scheme", type=str, default='exp', help="Determines the weighting technique used. Currently existing schemes are dirac, linear, and exp.")
parser.add_argument("--max_samples", type=int, default=5000, help="Temporary value that determines the maximum number of samples should be with each class.")
opt = parser.parse_args()
opt.n_epochs *= int((1-opt.frac_workers)*10)		#This is to cope up with the workers that remain idle in fl rounds...to achieve fair comparison with the single-machine implementation
print(opt)
port = opt.port
master = opt.master
#DIST
size = opt.size
rank = opt.rank
model = opt.model
assert opt.iid == 0 or opt.iid == 1
import socket
hostname = socket.gethostname()
if hostname == master:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if rank==0 else str((rank%2) + 1)		#%1 should be replaced by %(num_gpus-1)...now we are testing with 2 GPUs per machine
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%2) 			#Other machines can use both GPUs freely..only the master is allowed to take one GPU exclusively

cuda = True if torch.cuda.is_available() else False
print("Using Cuda?\n ", cuda, "Hostname: ", hostname)

init_processes(rank,size, run)
