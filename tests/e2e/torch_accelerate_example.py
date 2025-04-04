from accelerate import Accelerator
import torch
from torch import nn
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    # Initialize the accelerator
    accelerator = Accelerator()

    # Print the device and process rank
    print(f"Device: {accelerator.device}")
    print(f"Process rank: {accelerator.process_index}")

    # Test allreduce

    global_rank = accelerator.process_index
    world_size = accelerator.num_processes

    tensor = torch.tensor([global_rank + 1], device=accelerator.device)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Global rank {global_rank} reduced tensor: {tensor.item()}")

    assert (
        tensor.item() == world_size * (world_size + 1) // 2
    ), f"Expected {world_size * (world_size + 1) // 2}, got {tensor.item()}"

    # Example of using the accelerator for distributed training

    torch.manual_seed(0)
    model = TestModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    dataset = TestDataset(100)
    dataloader = DataLoader(dataset, batch_size=16)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    device = accelerator.device

    # Training loop
    for epoch in range(2):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs.to(device))
            optimizer.zero_grad()
            loss = nn.MSELoss()(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    main()
