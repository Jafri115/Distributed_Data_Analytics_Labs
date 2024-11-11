train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform)

batch_size = 60
shuffle_dataset = False
random_seed= 31142
num_epochs = 5

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)

indices = list(range(dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
testSplitRatio = 10000/dataset_size
validationSplitRatio = 10000/dataset_size
trainSplitRation = 1-testSplitRatio-validationSplitRatio

train_indices = indices[0:math.floor(dataset_size*trainSplitRation)]
val_indices = indices[math.floor(dataset_size*trainSplitRation):math.floor(dataset_size*(trainSplitRation+validationSplitRatio))]
test_indices = indices[math.floor(dataset_size*(trainSplitRation+validationSplitRatio)):]


train_sampler = SequentialSampler(train_indices)
valid_sampler = SequentialSampler(val_indices)
test_sampler = SequentialSampler(test_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           sampler=train_sampler,shuffle=False)
validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=valid_sampler,shuffle=False)
test_loader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=test_sampler,shuffle=False)