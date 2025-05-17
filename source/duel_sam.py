# In a simple setting compare traditional SAM vs Dueling SAM (replacing the gradient with dueling feedback)

import torch
import torch.optim as optim
from models import ResNet18
from utils import get_cifar_data, train, test
from tqdm import tqdm
import os
from datetime import datetime

class SAM(optim.Optimizer):
    """
    The SAM optimization problem can be written as: min_w max_{||ε||₂ ≤ ρ} L(w + ε).
    where:
    w are the model parameters
    L is the loss function
    ρ is the neighborhood size
    ε is the perturbation vector.

    The solution involves two steps:
        1. Find the worst-case perturbation ε.
        2. Update the parameters using this perturbation.

    The initialization:
        - Takes a base optimizer (like SGD)
        - Sets the neighborhood size ρ 
        - Initializes the base optimizer with the same parameters
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    """
    This implements the first step of SAM:
    
    1. Calculates the gradient norm
    2. For each parameter group:
        a. Computes the scale factor as ρ/||∇L(w)||₂
        b. Calculates the perturbation ε = ρ * ∇L(w)/||∇L(w)||₂
        c. Adds the perturbation to the parameters
        d. Stores the perturbation for the second step
    """
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # numerical stability with the 1e-12 term in the denominator
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                # Math: ε = ρ * ∇L(w)/||∇L(w)||₂
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    """
    This implements the second step:
        1. Removes the perturbation from the parameters
        2. Performs the actual parameter update using the base optimizer
    """
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                # Math: w = w - η * ∇L(w + ε)
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

def main(use_sam=True):
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}_sam{use_sam}"
    run_dir = os.path.join(outputs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
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
    criterion = torch.nn.CrossEntropyLoss()
    
    if use_sam:
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nTraining Progress ({'SAM' if use_sam else 'Regular'}):")
    pbar = tqdm(total=epochs, position=0)
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        pbar.update(1)
        scheduler.step()
    
    pbar.close()
    
    # Save results
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'hyperparameters': {
            'dataset': dataset,
            'batch_size': batch_size,
            'epochs': epochs,
            'rho': rho,
            'learning_rate': learning_rate,
            'use_sam': use_sam
        }
    }
    
    # Save model and results
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pth'))
    torch.save(results, os.path.join(run_dir, 'results.pt'))
    
    print(f"\nResults saved in: {run_dir}")

if __name__ == '__main__':
    # You can change use_sam to False for regular training
    main(use_sam=False)