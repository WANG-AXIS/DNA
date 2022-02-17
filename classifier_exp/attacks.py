import torch
##Includes functions for all adversarial attack algorithms
#attacks are based on cross entropy criterion

loss_ce = torch.nn.CrossEntropyLoss()

def FGSM_linf(x, y, model, epsilon):
    model.eval()
    model.zero_grad()
    x.requires_grad = True
    y_hat = model(x)
    loss = loss_ce(y_hat, y)
    loss.backward()
    perturbed_x = torch.clamp(x + epsilon*(x.grad.data).sign(), min=0, max=1.0)
    return perturbed_x

def FGSM_l2(x, y, model, epsilon):
    epsilon = epsilon*28
    model.eval()
    model.zero_grad()
    x.requires_grad = True
    y_hat = model(x)
    loss = loss_ce(y_hat, y)
    loss.backward()
    perturbed_x = torch.clamp(x + epsilon*x.grad.data/torch.linalg.norm(x.grad.data), min=0, max=1.0)
    return perturbed_x


def PGD_linf(x, y, model, epsilon, k=40, a=3/255):
    model.eval()
    device = torch.device("cuda:{}".format(x.get_device()) if x.is_cuda else "cpu")
    perturbed_x = x + (2*epsilon*torch.rand(x.shape) - epsilon).to(device)
    perturbed_x = torch.clamp(perturbed_x, 0, 1)
    for i in range(k):
        perturbed_x.requires_grad = True
        model.zero_grad()
        loss = loss_ce(model(perturbed_x),y)
        loss.backward()
        perturbed_x = perturbed_x + a*perturbed_x.grad.data.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        perturb = torch.clamp(perturbed_x-x, -epsilon, epsilon)
        perturbed_x = (x + perturb).detach()
    return perturbed_x

def PGD_l2(x, y, model, epsilon, k=40, a=3/255):
    epsilon = epsilon*28 # hardcode dimension adjustment
    model.eval()
    device = torch.device("cuda:{}".format(x.get_device()) if x.is_cuda else "cpu")
    perturbed_x = x + (2*epsilon*torch.rand(x.shape) - epsilon).to(device)
    perturbed_x = torch.clamp(perturbed_x, 0, 1)
   # print(torch.linalg.norm(perturbed_x-x))
    for i in range(k):
        perturbed_x.requires_grad = True
        model.zero_grad()
        loss = loss_ce(model(perturbed_x),y)
        loss.backward()
        perturbed_x = perturbed_x + a*perturbed_x.grad.data
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        perturb = perturbed_x-x
        norm = torch.linalg.norm(perturb, dim=(1,2,3))
        perturb = (norm<=epsilon).reshape([-1,1,1,1])*perturb + (epsilon/norm*(norm>epsilon)).reshape([-1,1,1,1])*perturb
        perturbed_x = (x + perturb).detach()
    return perturbed_x