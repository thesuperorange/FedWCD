import numpy as np
from .utils import *
from lapsolver import solve_dense

import torch

    
def compute_cost(global_atoms, atoms_j, global_atoms_squared, sigma, sigma0, mu0, popularity_counts, gamma, J):
    
    Lj = atoms_j.shape[0]
    counts = np.array(popularity_counts).astype(np.float32)
    sigma_ratio = (sigma0/sigma).astype(np.float32)
    denum_match = (np.outer(counts+1,sigma0) + sigma).astype(np.float32)
    logger.info("global_atoms type: {}, atoms_j type: {}, denum_match type: {}".format(global_atoms.dtype, atoms_j.dtype, denum_match.dtype))
    
    atoms_j_cuda = torch.from_numpy(atoms_j).to('cuda:0')
    global_atoms_cuda = torch.from_numpy(global_atoms).to('cuda:0')
    denum_match_cuda = torch.from_numpy(denum_match).to('cuda:0')
    sigma_ratio_cuda = torch.from_numpy(sigma_ratio).to('cuda:0')
    mu0_cuda = torch.from_numpy(mu0).to('cuda:0')

    #param_cost_cuda = torch.stack([((sigma_ratio_cuda*(atoms_j_cuda[l]+global_atoms_cuda)**2 + 2*mu0_cuda*(atoms_j_cuda[l]+global_atoms_cuda))/denum_match_cuda).sum(axis=1) for l in range(Lj)])
    param_cost_cuda = torch.stack([((sigma_ratio_cuda*(atoms_j_cuda[l]+global_atoms_cuda)**2 + 2*mu0_cuda*(atoms_j_cuda[l]+global_atoms_cuda))/denum_match_cuda).sum(dim=1) for l in range(Lj)])
    
    param_cost = param_cost_cuda.to('cpu').numpy()
    logger.info("param_cost shape: {}".format(param_cost.shape))

    denum_no_match = np.outer(counts,sigma0) + sigma
    cost_no_match = ((sigma_ratio*(global_atoms)**2 + 2*mu0*(global_atoms))/denum_no_match).sum(axis=1)
    sigma_cost = (np.log(denum_no_match) - np.log(denum_match)).sum(axis=1)
    mu_cost = (np.outer(counts,mu0**2)/denum_no_match - np.outer(counts+1,mu0**2)/denum_match).sum(axis=1)

    # cost_no_match = ((sigma_ratio*(global_atoms)**2 + 2*mu0*(global_atoms))/denum_no_match).sum(dim=1)
    # sigma_cost = (np.log(denum_no_match) - np.log(denum_match)).sum(dim=1)
    # mu_cost = (np.outer(counts,mu0**2)/denum_no_match - np.outer(counts+1,mu0**2)/denum_match).sum(dim=1)

    counts = np.minimum(counts,10)
    param_cost = np.array(param_cost) - cost_no_match + sigma_cost + mu_cost + 2*np.log(counts/(J-counts))
    
    ## Nonparametric cost
    # max_added = min(Lj, max(700 - L, 1))
    #max_added = Lj
    L = global_atoms.shape[0]
    max_added = min(Lj, max(2*Lj - L, 1))

    nonparam_cost = ((sigma_ratio*atoms_j**2 + 2*mu0*atoms_j - mu0**2)/(sigma0+sigma)).sum(axis=1)
    #nonparam_cost = ((sigma_ratio*atoms_j**2 + 2*mu0*atoms_j - mu0**2)/(sigma0+sigma)).sum(dim=1)
    nonparam_cost = np.outer(nonparam_cost, np.ones(max_added))
    cost_pois = 2*np.log(np.arange(1,max_added+1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2*np.log(gamma/J)
    
    ## sigma penalty
    nonparam_cost += np.log(sigma).sum() - np.log(sigma0+sigma).sum()
    
    full_cost = np.hstack((param_cost, nonparam_cost)).astype(np.float32)
    return full_cost


def matching_upd_j(atoms_j, global_atoms, global_atoms_squared, sigma, sigma0, mu0, popularity_counts, gamma, J):
    
    L = global_atoms.shape[0]
    
    compute_cost_start = time.time()
    full_cost = compute_cost(global_atoms.astype(np.float32), atoms_j.astype(np.float32), global_atoms_squared.astype(np.float32), sigma, sigma0, mu0.astype(np.float32), popularity_counts, gamma, J)
    compute_cost_dur = time.time() - compute_cost_start
    logger.info("Compute cost duration: {}".format(compute_cost_dur))

    #row_ind, col_ind = linear_sum_assignment(-full_cost)
    row_ind, col_ind = solve_dense(-full_cost)

    assignment_j = []
    
    new_L = L
    
    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_atoms[i] += atoms_j[l]
            global_atoms_squared[i] += atoms_j[l]**2
        else:
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_atoms = np.vstack((global_atoms,atoms_j[l]))
            global_atoms_squared = np.vstack((global_atoms_squared,atoms_j[l]**2))

    return global_atoms, global_atoms_squared, popularity_counts, assignment_j

def objective(global_atoms, popularity_counts, sigma, sigma0, mu0):
    popularity_counts = np.copy(popularity_counts)
    obj_denum = np.outer(popularity_counts, sigma0) + sigma
    obj_num = (sigma0/sigma)*global_atoms**2 + 2*mu0*global_atoms - np.outer(popularity_counts,mu0**2)
    obj = (obj_num/obj_denum - np.log(obj_denum)).sum()
    return obj
    
def hyperparameters(global_atoms, global_atoms_squared, popularity_counts):
    popularity_counts = np.copy(popularity_counts)
    mean_atoms = global_atoms/popularity_counts.reshape(-1,1)
    mu0 = mean_atoms.mean(axis=0)
    sigma = global_atoms_squared - (global_atoms**2)/popularity_counts.reshape(-1,1)
    sigma = sigma.sum(axis=0)/(popularity_counts.sum() - len(popularity_counts))
    sigma = np.maximum(sigma,1e-10)
    sigma0 = ((mean_atoms - mu0)**2).mean(axis=0)
    sigma0 = sigma0 - sigma*((1/popularity_counts).sum())/len(popularity_counts)
    sigma0 = np.maximum(sigma0,1e-10)
    return mu0, sigma, sigma0
    
def match_local_atoms(local_atoms, sigma, sigma0, gamma, it, optimize_hyper=True):
    J = len(local_atoms)
    D = local_atoms[0].shape[1]
    
    group_order = sorted(range(J), key = lambda x: -local_atoms[x].shape[0])
    
    sigma = np.ones(D)*sigma
    sigma0 = np.ones(D)*sigma0
    total_atoms = sum([atoms_j.shape[0] for atoms_j in local_atoms])
    mu0 = sum([atoms_j.sum(axis=0) for atoms_j in local_atoms])/total_atoms
    logger.info('Init mu0 estimate mean is %f' % (mu0.mean()))
    
    global_atoms = np.copy(local_atoms[group_order[0]])
    global_atoms_squared = np.copy(local_atoms[group_order[0]]**2)
    
    popularity_counts = [1]*global_atoms.shape[0]
    
    assignment = [[] for _ in range(J)]
    
    assignment[group_order[0]] = list(range(global_atoms.shape[0]))
    
    ## Initialize
    for j in group_order[1:]:
        global_atoms, global_atoms_squared, popularity_counts, assignment_j = matching_upd_j(local_atoms[j], global_atoms, global_atoms_squared, sigma, sigma0, mu0, popularity_counts, gamma, J)
        assignment[j] = assignment_j
    
    if optimize_hyper:
        mu0, sigma, sigma0 = hyperparameters(global_atoms, global_atoms_squared, popularity_counts)
        logger.info('Init Sigma mean estimate is %f; sigma0 is %f; mu0 is %f' % (sigma.mean(),sigma0.mean(),mu0.mean()))
    
    logger.info('Init objective (without prior) is %f; number of global atoms is %d' % (objective(global_atoms, popularity_counts, sigma, sigma0, mu0),global_atoms.shape[0]))
    
    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order: #random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj),assignment[j]), key = lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                print('Warning - weird unmatching')
                else:
                    global_atoms[i] = global_atoms[i] - local_atoms[j][l]
                    global_atoms_squared[i] = global_atoms_squared[i] - local_atoms[j][l]**2
                    
            global_atoms = np.delete(global_atoms,to_delete,axis=0)
            global_atoms_squared = np.delete(global_atoms_squared,to_delete,axis=0)
    
            ## Match j
            global_atoms, global_atoms_squared, popularity_counts, assignment_j = matching_upd_j(local_atoms[j], global_atoms, global_atoms_squared, sigma, sigma0, mu0, popularity_counts, gamma, J)
            assignment[j] = assignment_j
        
        if optimize_hyper:
            mu0, sigma, sigma0 = hyperparameters(global_atoms, global_atoms_squared, popularity_counts)
            logger.info('Sigma mean estimate is %f; sigma0 is %f; mu0 is %f' % (sigma.mean(),sigma0.mean(),mu0.mean()))
            
        logger.info('Matching iteration %d' % iteration)
        logger.info('Objective (without prior) at iteration %d is %f; number of global atoms is %d' % (iteration,objective(global_atoms, popularity_counts, sigma, sigma0, mu0),global_atoms.shape[0]))
    
    logger.info('Number of global atoms is %d, gamma %f' % (global_atoms.shape[0], gamma))
    
    map_out = (mu0*sigma + global_atoms*sigma0)/(np.outer(popularity_counts,sigma0)+sigma)
    return assignment, map_out, popularity_counts, (mu0.mean(), sigma.mean(), sigma0.mean())