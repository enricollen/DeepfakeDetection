import torch

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples_class_0=100, num_samples_class_1=100):
        self.num_samples_class_0 = num_samples_class_0
        self.num_samples_class_1 = num_samples_class_1

        self.labels = torch.cat([
            torch.zeros(num_samples_class_0, dtype=torch.long),
            torch.ones(num_samples_class_1, dtype=torch.long)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, self.labels[idx]

class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.label_to_indices = {}
        for i in range(len(dataset)):
            label = int(dataset.labels[i].item()) 
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        self.minority_class = min(self.label_to_indices)
        self.majority_class = max(self.label_to_indices)
        self.minority_indices = self.label_to_indices[self.minority_class]
        self.majority_indices = self.label_to_indices[self.majority_class]

        # Track the number of samples seen from class 1
        self.num_samples_seen_class_1 = 0

    def __iter__(self):
        num_minority = len(self.minority_indices)
        num_majority = len(self.majority_indices)

        num_samples = min(num_minority, num_majority)
        minority_pointer = 0
        majority_pointer = 0

        for _ in range(num_samples):
            yield self.minority_indices[minority_pointer]
            yield self.majority_indices[majority_pointer]

            minority_pointer += 1
            majority_pointer += 1

            # number of samples seen from class 1 incremented
            self.num_samples_seen_class_1 += 1

            # If all samples from class 1 have been seen, exit the iteration
            if self.num_samples_seen_class_1 == len(self.majority_indices):
                print("All the samples from class 1 have been seen, going to next epoch.")
                return

        if num_minority > num_majority:
            remaining_indices = self.minority_indices[minority_pointer:]
        else:
            remaining_indices = self.majority_indices[majority_pointer:]

        for index in remaining_indices:
            yield index

    def __len__(self):
        return len(self.dataset)


# Simulation of epochs
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}:")

    toy_dataset = ToyDataset(num_samples_class_0=80, num_samples_class_1=20)
    balanced_sampler = BalancedSampler(toy_dataset)

    # DataLoader with BalancedSampler
    batch_size = 13
    data_loader = torch.utils.data.DataLoader(toy_dataset, batch_size=batch_size, sampler=balanced_sampler)

    for batch_idx, (indices, labels) in enumerate(data_loader):
        class_0_count = torch.sum(labels == 0).item()
        class_1_count = torch.sum(labels == 1).item()
        total_samples = len(labels)

        print(f"  Batch {batch_idx + 1}: Class 0 count - {class_0_count}, Class 1 count - {class_1_count}, Total samples - {total_samples}")
