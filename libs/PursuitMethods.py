#title MachingPursuit
import torch

def MP(input, dictionary, max_iterations=10):
    representation = torch.zeros(dictionary.shape[0], input.shape[1])
    residual = input.clone()
    for i in range(max_iterations):
        # Find the dictionary atom with the largest inner product with the residual
        inner_products = torch.matmul(dictionary.t(), residual)
        max_inner_product, max_atom = torch.max(torch.abs(inner_products), dim=0)
        # Update the representation
        representation[max_atom, :] = torch.matmul(dictionary[max_atom, :].view(1, -1), residual) / (max_inner_product + 1e-6)
        # Update the residual
        residual = input - torch.matmul(dictionary, representation)
    return representation

def OMP(input, dictionary, max_iterations=10):
    representation = torch.zeros(input.shape[0],dictionary.shape[0])
    residual = input.clone()
    for j in range(input.size(0)):
        active_set = []
        for i in range(max_iterations):
            # Find the dictionary atom with the largest inner product with the residual
            inner_products = torch.matmul(residual[j],dictionary.t())
            max_inner_product, max_atom = torch.max(torch.abs(inner_products), dim=0)
            # Update the active set
            active_set.append(max_atom.item())
            # Compute the weights that minimize the L2 norm of the residual
            weights = torch.linalg.lstsq(dictionary[active_set, :].t(), residual[j], rcond=None)[0]
            # Update the representation
            representation[j,active_set] = weights #torch.matmul(weights.view(-1,1),input[j].view(1,-1))
            # Update the residual
            residual[j] = input[j] - torch.matmul(representation[j],dictionary)
    #errorinmat = input - torch.matmul(representation, dictionary)    
    #errorinmat = torch.linalg.norm(errorinmat,dim=1)
    #print(errorinmat.mean())
    return representation


'''def OMP(input, dictionary, max_iterations=10):
    representation = torch.zeros(dictionary.shape[0], input.shape[1])
    residual = input.clone()
    active_set = []
    for i in range(max_iterations):
        # Find the dictionary atom with the largest inner product with the residual
        inner_products = torch.matmul(residual,dictionary.t())
        max_inner_product, max_atom = torch.max(torch.abs(inner_products), dim=0)
        # Update the active set
        active_set.append(max_atom.item())
        # Compute the weights that minimize the L2 norm of the residual
        weights = torch.lstsq(dictionary[active_set, :], residual, rcond=None)[0]
        # Update the representation
        representation[active_set, :] += torch.matmul(weights, input)
        # Update the residual
        residual = input - torch.matmul(dictionary, representation)
    return representation'''