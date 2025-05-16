# In a simple setting compare traditional SAM vs Dueling SAM (replacing the gradient with dueling feedback)

import torch
import torch.optim as optim
from models import ResNet18
from utils import get_cifar_data, train, test
from tqdm import tqdm

class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

def main():
    # Hyperparameters
    dataset = 'cifar100'  # or 'cifar100'
    batch_size = 256
    epochs = 200
    rho = 0.05
    learning_rate = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data
    trainloader, testloader, num_classes = get_cifar_data(dataset, batch_size)
    
    # Initialize model
    model = ResNet18(num_classes=num_classes).to(device)
    
    # Initialize optimizer and criterion
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs)
    
    print("\nTraining Progress:")
    pbar = tqdm(total=epochs, position=0)
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        pbar.update(1)
        scheduler.step()
    
    pbar.close()

if __name__ == '__main__':
    main()