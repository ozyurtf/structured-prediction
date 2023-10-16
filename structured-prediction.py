import torch.optim as optim
from utils import *
import copy
import torch

simple_transforms = transforms.Compose([transforms.ToTensor()])

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

model = SimpleNet()
alphabet_energies = model(alphabet.view(1, 1, *alphabet.shape))
    
plot_energies(alphabet_energies[0].detach())

sds = SimpleWordsDataset(1, len=1000, jitter=True, noise=False)
dataloader = torch.utils.data.DataLoader(sds, batch_size=16, num_workers=0, collate_fn=simple_collate_fn)

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_model(model = model, 
            epochs = 50, 
            dataloader = dataloader, 
            criterion = cross_entropy, 
            optimizer = optimizer)
   
tds = SimpleWordsDataset(1, len=100)
get_accuracy(model, tds)

alphabet_energies_post_train = model(alphabet.view(1, 1, *alphabet.shape))
plot_energies(alphabet_energies_post_train[0].detach())

characters_with_min_energy = [string.ascii_lowercase[val] for val in alphabet_energies_post_train[0][0].argmin(dim=1)]

print("The characters that have the least energy in each window: '{}'".format(''.join(characters_with_min_energy)))

energies = model(alphabet.view(1, 1, *alphabet.shape))
targets = transform_word(string.ascii_lowercase).unsqueeze(0)

energies = energies.squeeze(1)
pm = build_path_matrix(energies, targets)
plot_pm(pm[0].detach())

characters_with_min_energy = [string.ascii_lowercase[val] for val in energies[0].argmin(dim=1)]

print("The characters that have the least energy in each window: '{}'".format(''.join(characters_with_min_energy)))

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

free_energy, path, d = find_path(pm[0])
plot_pm(pm[0].detach(), path)
print('Free energy is', free_energy.item())

plt.figure(dpi=200)
plt.imshow(d.cpu().detach().T.clamp(torch.min(d).item(), 1000))
plt.axis('off')
    
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
    
min_indices = energies[0].argmin(dim=-1)
print(indices_to_str(min_indices, energies))




