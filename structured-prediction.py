from PIL import ImageDraw, ImageFont
import string
import random
import torch
import torchvision
from torchvision import transforms
from PIL import Image 
from matplotlib import pyplot as plt
import gtn
from torch import nn
from collections import Counter
import copy
import time
import numpy as np
import torch.optim as optim
from tqdm import tqdm

simple_transforms = transforms.Compose([transforms.ToTensor()])

class SimpleWordsDataset(torch.utils.data.IterableDataset):
  def __init__(self, max_length, len=100, jitter=False, noise=False):
    self.max_length = max_length
    self.transforms = transforms.ToTensor()
    self.len = len
    self.jitter = jitter
    self.noise = noise
  
  def __len__(self):
    return self.len

  def __iter__(self):
    for _ in range(self.len):
        text = ''.join([random.choice(string.ascii_lowercase) for i in range(self.max_length)])
        img = self.draw_text(text, jitter=self.jitter, noise=self.noise)
        yield img, text
  
  def draw_text(self, text, length=None, jitter=False, noise=False):
    if length == None:
        length = 18 * len(text)
    img = Image.new('L', (length, 32))
    fnt = ImageFont.truetype("fonts/Anonymous.ttf", 20)

    d = ImageDraw.Draw(img)
    pos = (0, 5)
    if jitter:
        pos = (random.randint(0, 7), 5)
    else:
        pos = (0, 5)
    d.text(pos, text, fill=1, font=fnt)

    img = self.transforms(img)
    img[img > 0] = 1 
    
    if noise:
        img += torch.bernoulli(torch.ones_like(img) * 0.1)
        img = img.clamp(0, 1)
        

    return img[0]

sds = SimpleWordsDataset(1, jitter=True, noise=False)
img = next(iter(sds))[0]
print(img.shape)
plt.imshow(img)

fig, ax = plt.subplots(3, 9, figsize=(12, 6), dpi=200)

for i, c in enumerate(string.ascii_lowercase):
    row = i // 9
    col = i % 9
    ax[row][col].imshow(sds.draw_text(c))
    ax[row][col].axis('off')
ax[2][8].axis('off')
    
plt.show()

alphabet = sds.draw_text(string.ascii_lowercase, 340)
plt.figure(dpi=200)
plt.imshow(alphabet)
plt.axis('off')

ALPHABET_SIZE = 27
BETWEEN = 26

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 1,
                               kernel_size = (4,8))  
        self.relu1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = (3,6))
        self.linear1 = nn.Linear(9, 18)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(18, 27) 
        
    def forward(self, x):
        x = self.conv1(x)  
        x = self.relu1(x)
        x = self.max_pool(x)
        x = x.permute(0, 1, 3, 2)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

model = SimpleNet()
alphabet_energies = model(alphabet.view(1, 1, *alphabet.shape))

def plot_energies(ce):
    fig=plt.figure(dpi=200)
    ax = plt.axes()
    im = ax.imshow(ce.cpu().T)
    
    ax.set_xlabel('window locations →')
    ax.set_ylabel('← classes')
    ax.xaxis.set_label_position('top') 
    ax.set_xticks([])
    ax.set_yticks([])
    
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax) 
    
plot_energies(alphabet_energies[0].detach())

def train_model(model, epochs, dataloader, criterion, optimizer):
    for epoch in tqdm(range(epochs)): 
      model.train()
      for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()

        energies = model(inputs)
        energies = energies.squeeze(1)
        energies = energies[:, 0, :]

        loss = criterion(energies, labels)

        loss.backward()
        optimizer.step() 

def cross_entropy(energies, *args, **kwargs):
    """ We use energies, and therefore we need to use log soft arg min instead
        of log soft arg max. To do that we just multiply energies by -1. """
    return nn.functional.cross_entropy(-1 * energies, *args, **kwargs)

def simple_collate_fn(samples):
    images, annotations = zip(*samples)
    images = list(images)
    annotations = list(annotations)
    annotations = list(map(lambda c : torch.tensor(ord(c) - ord('a')), annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
        
    if len(images) == 1:
        return images[0].unsqueeze(0), torch.stack(annotations)
    else:
        return torch.stack(images), torch.stack(annotations)

sds = SimpleWordsDataset(1, len=1000, jitter=True, noise=False)
dataloader = torch.utils.data.DataLoader(sds, batch_size=16, num_workers=0, collate_fn=simple_collate_fn)

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_model(model = model, 
            epochs = 50, 
            dataloader = dataloader, 
            criterion = cross_entropy, 
            optimizer = optimizer)

def get_accuracy(model, dataset):
    cnt = 0
    for i, l in dataset:
        energies = model(i.unsqueeze(0).unsqueeze(0))[0, 0]
        x = energies.argmin(dim=-1)
        cnt += int(x == (ord(l[0]) - ord('a')))
    return cnt / len(dataset)
        
tds = SimpleWordsDataset(1, len=100)
get_accuracy(model, tds)

alphabet_energies_post_train = model(alphabet.view(1, 1, *alphabet.shape))
plot_energies(alphabet_energies_post_train[0].detach())

characters_with_min_energy = [string.ascii_lowercase[val] for val in alphabet_energies_post_train[0][0].argmin(dim=1)]

print("The characters that have the least energy in each window: '{}'".format(''.join(characters_with_min_energy)))

def build_path_matrix(energies, targets):
    # inputs: 
    #    energies, shape is BATCH_SIZE x L x 27
    #    targets, shape is BATCH_SIZE x T
    # 
    # outputs:
    #    a matrix of shape BATCH_SIZE x L x T
    #    where output[i, j, k] = energies[i, j, targets[i, k]]
    #
    
    batch_size, L, _ = energies.shape
    T = targets.shape[1]
    expanded_targets = targets.unsqueeze(1).expand(batch_size, L, T)
    path_matrix = energies.gather(2, expanded_targets)
    
    return path_matrix   

def build_ce_matrix(energies, targets):
    # inputs: 
    #   energies, shape is BATCH_SIZE x L x 27
    #   targets, shape is BATCH_SIZE x T
    # L is \ververtt = targets.unsqueeze(1).repeat(1,energies.shape[1],1)t l \vert
    # T is \vert y \vert
    
    # outputs:
    #   a matrix ce of shape BATCH_SIZE x L x T
    #   where ce[i, j, k] = cross_entropy(energies[i, j], targets[i, k])
    
    batch_size = energies.shape[0]
    L = energies.shape[1]
    T = targets.shape[1]
    cross_entropies = torch.empty(batch_size, L, T)
    for i in range(energies.shape[0]): 
      for j in range(energies.shape[1]): 
        for k in range(targets.shape[1]):
          cross_entropies[i][j][k] = nn.functional.cross_entropy(energies[i][j], targets[i][k])

    return cross_entropies

def transform_word(s):
    # input: a string
    # output: a tensor of shape 2*len(s)
    transformed_str = '_'.join(s) + '_'

    char_to_index = {char: index for index, char in enumerate('abcdefghijklmnopqrstuvwxyz_')}
    transformed_tensor = torch.tensor([char_to_index[char] for char in transformed_str], dtype=torch.long)

    return transformed_tensor    

def plot_pm(pm, path=None):
    fig=plt.figure(dpi=200)
    ax = plt.axes()
    im = ax.imshow(pm.cpu().T)
    
    ax.set_xlabel('window locations →')
    ax.set_ylabel('← label characters')
    ax.xaxis.set_label_position('top') 
    ax.set_xticks([])
    ax.set_yticks([])
    
    if path is not None:
        for i in range(len(path) - 1):
            ax.plot(*path[i], *path[i+1], marker = 'o', markersize=0.5, linewidth=10, color='r', alpha=1)

    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax) 

energies = model(alphabet.view(1, 1, *alphabet.shape))
targets = transform_word(string.ascii_lowercase).unsqueeze(0)

energies = energies.squeeze(1)
pm = build_path_matrix(energies, targets)
plot_pm(pm[0].detach())

characters_with_min_energy = [string.ascii_lowercase[val] for val in energies[0].argmin(dim=1)]

print("The characters that have the least energy in each window: '{}'".format(''.join(characters_with_min_energy)))

def path_energy(pm, path):
    # inputs:
    #   pm - a matrix of energies 
    #    L - energies length
    #    T - targets length
    #   path - list of length L that maps each energy vector to an element in T 
    # returns:
    #   energy - sum of energies on the path, or 2**30 if the mapping is invalid
    path_energies = []
    points = list(zip(range(pm.shape[0]), path))
    for p in points:
      path_energies.append(pm[p])
    sum_of_energies = sum(path_energies)
    
    return sum_of_energies

path = torch.zeros(energies.shape[1] - 1)
path[:targets.shape[1] - 1] = 1
path = [0] + list(map(lambda x : x.int().item(), path[torch.randperm(path.shape[0])].cumsum(dim=-1)))
points = list(zip(range(energies.shape[1]), path))

plot_pm(pm[0].detach(), points)
print('Energy is', path_energy(pm[0], path).item())


bad_path_1 = [1]*(energies.shape[1])
bad_points_1 = list(zip(range(energies.shape[1]), bad_path_1))
plot_pm(pm[0].detach(), bad_points_1)
print('Energy is', path_energy(pm[0], bad_path_1).item())

dp = torch.zeros(pm[0].shape)

def find_path(pm):
    # inputs:
    #   pm - a tensor of shape LxT with energies
    #     L is length of energies array
    #     T is target sequence length
    # NOTE: this is slow because it's not vectorized to work with batches.
    #  output:
    #     a tuple of three elements:
    #         1. sum of energies on the best path,
    #         2. list of tuples - points of the best path in the pm matrix 
    #         3. the dp array

    L, T = pm.shape
    
    dp = torch.zeros((L, T))
    path = torch.zeros((L, T), dtype=torch.long)
    
    dp[0, 0] = pm[0, 0]
    
    for i in range(1, L):
      dp[i, 0] = dp[i - 1, 0] + pm[i, 0]
      path[i, 0] = 1  
    
    for j in range(1, T):
      dp[0, j] = dp[0, j - 1] + pm[0, j]
      path[0, j] = 2 
    
    for i in range(1, L):
      for j in range(1, T):
        if dp[i - 1, j] < dp[i - 1, j - 1]:
          dp[i, j] = dp[i - 1, j] + pm[i, j]
          path[i, j] = 1  
        else:
          dp[i, j] = dp[i - 1, j - 1] + pm[i, j]
          path[i, j] = 0 
    
    optimal_path = []
    i, j = L - 1, T - 1
    while len(optimal_path) < max(L, T):
      optimal_path.append((i, j))
      if path[i, j] == 1:
        i -= 1
      elif path[i, j] == 2:
        j -= 1
      else:
        i -= 1
        j -= 1
    
    optimal_path.reverse()
    
    return dp[L-1, T-1], optimal_path, dp


free_energy, path, d = find_path(pm[0])
plot_pm(pm[0].detach(), path)
print('Free energy is', free_energy.item())

plt.figure(dpi=200)
plt.imshow(d.cpu().detach().T.clamp(torch.min(d).item(), 1000))
plt.axis('off')

def train_ebm_model(model, num_epochs, train_loader, criterion, optimizer):
    ''' Train EBM Model using find_path()'''
    pbar = tqdm(range(num_epochs))
    total_train_loss = 0.0
    size = 0
    free_energies = []
    paths = []
    model.train()
    for epoch in pbar:
      start_time = time.time()
      for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.unsqueeze(1)
        ebm_optimizer.zero_grad()
        energies = model(inputs)
        energies = energies.squeeze(1)
        ebm_pm_matrix = build_path_matrix(energies, labels)
        for val in range(energies.shape[0]):
          ebm_pm_matrix_i = ebm_pm_matrix[val]
          ebm_free_energy_i, ebm_path_i, ebm_d_i = find_path(ebm_pm_matrix_i)
          ebm_path_i_tensor = torch.tensor(ebm_path_i)

          label_i = labels[val].detach()
          energy_i = ebm_pm_matrix_i[ebm_path_i_tensor[:, 0], 
                                     ebm_path_i_tensor[:, 1]].detach()

          energy_i.requires_grad = True
          ce = criterion(energy_i.float(), label_i.float())
          ce.backward()
          optimizer.step() 
          total_train_loss += ce.item()

      epoch_time = time.time() - start_time
      pbar.set_postfix({'train_loss': total_train_loss / len(sds), 'Epoch Time': epoch_time}) 

def collate_fn(samples):
    """ A function to collate samples into batches for multi-character case"""
    images, annotations = zip(*samples)
    images = list(images)
    annotations = list(annotations)
    annotations = list(map(transform_word, annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    m_length = max(3, max([s.shape[0] for s in annotations]))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
        annotations[i] = torch.nn.functional.pad(annotations[i], (0, m_length - annotations[i].shape[0]), value=BETWEEN)
    if len(images) == 1:
        return images[0].unsqueeze(0), torch.stack(annotations)
    else:
        return torch.stack(images), torch.stack(annotations)
    
sds = SimpleWordsDataset(2, 2500) 

BATCH_SIZE = 32
dataloader = torch.utils.data.DataLoader(sds, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate_fn)

ebm_model = copy.deepcopy(model)
ebm_optimizer = optim.Adam(ebm_model.parameters(), lr = 0.001)

train_ebm_model(model = ebm_model, 
                num_epochs = 20, 
                train_loader = dataloader, 
                criterion = cross_entropy,
                optimizer = optimizer)

energies = ebm_model(alphabet.unsqueeze(0).unsqueeze(0)).squeeze(1)
targets = transform_word(string.ascii_lowercase).reshape(1,52)
pm = build_path_matrix(energies, targets)

free_energy, path, _ = find_path(pm[0])
plot_pm(pm[0].detach(), path)
print('Free energy is', free_energy.item())

alphabet_energy_post_train_viterbi = ebm_model(alphabet.view(1, 1, *alphabet.shape))

plt.figure(dpi=200, figsize=(40, 10))
plt.imshow(alphabet_energy_post_train_viterbi.cpu().data[0].T)
plt.axis('off')


img = sds.draw_text('hello')
energies = ebm_model(img.unsqueeze(0).unsqueeze(0))
plt.imshow(img)
plot_energies(energies[0].detach().cpu())


def indices_to_str(indices):
    # inputs: indices - a tensor of most likely class indices
    # outputs: decoded string
    
    transformed_str = '_'.join(string.ascii_lowercase) + '_'

    char_to_index = {char: index for index, char in enumerate('abcdefghijklmnopqrstuvwxyz_')}
    index_to_char = {v: k for k, v in char_to_index.items()}

    indices = energies[0].argmin(dim=-1)
    indices = indices.tolist()[0]

    decoded_string = ''.join([index_to_char[val] for val in indices])   
    
    if '_' in decoded_string:
      most_common_characters = [Counter(val).most_common(1) for val in decoded_string.split('_')]

      final_string = ''
      for val in most_common_characters:
        if val != []:
          final_string += val[0][0]
        else: 
          final_string += ''
      return final_string
    else:
      return decoded_string    
    
min_indices = energies[0].argmin(dim=-1)
print(indices_to_str(min_indices))




