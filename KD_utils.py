import torch
import torch.nn.functional as F

def KD_getloss(Ps, Pt, Rs, Rt, yreg, yreg_t, LsL1, Lhard, iteration, pos_w, neg_w, mGPU, u=0.5, v=0.5, m=0.):

    
    
    #Lsoft(Ps,Pt)
    loss_soft_rpn = weighted_KL_div(Ps, Pt, pos_w, neg_w) # w0=1.5 for background, wi=1 for others
    
    #kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
    #kl = F.kl_div(Ps.log(), Pt, reduction='mean')

    
    # Lcls = u*Lhard(Ps,y)+(1-u)*Lsoft(Ps,Pt)
    loss_cls = u*Lhard+(1-u)*loss_soft_rpn

    #Lb(Rs,Rt,yreg)
    loss_b = bounded_regression_loss(Rs, Rt, yreg,yreg_t, m)            
    #Lreg = LsL1(Rs, yreg) + v* Lb(Rs,Rt,yreg)
    loss_reg = LsL1+ v*loss_b
    
    if iteration % 100==0:
        if mGPU:
            print('[KD_tuils] LsL1: %.4f' % (LsL1.mean().item()))
            print('[KD_tuils] Lhard: %.4f'% (Lhard.mean().item()))

            print('[KD_tuils] loss_soft: %.4f' % (loss_soft_rpn.mean().item()))
            #print('[KD_tuils] loss_soft F.KL: %.4f' % (kl.item()))

            print('[bounded_regression_loss] Rs v.s. gt: %.4f' % (F.mse_loss(Rs, yreg).mean().item()))
            print('[bounded_regression_loss] Rt v.s. gt: %.4f' % (F.mse_loss(Rt, yreg_t).mean().item()))
            print('[KD_tuils] loss_b: %.4f'%(loss_b.item()))            
            
        else:
            print('[KD_tuils] LsL1: %.4f' % (LsL1.mean().item()))
            print('[KD_tuils] Lhard: %.4f'% (Lhard.mean().item()))

            print('[KD_tuils] loss_soft: %.4f' % (loss_soft_rpn.mean().item()))
            #print('[KD_tuils] loss_soft F.KL: %.4f' % (kl.item()))

            print('[bounded_regression_loss] Rs v.s. gt: %.4f' % (F.mse_loss(Rs, yreg).item()))
            print('[bounded_regression_loss] Rt v.s. gt: %.4f' % (F.mse_loss(Rt, yreg_t).item()))
            print('[KD_tuils] loss_b: %.4f'%(loss_b.item()))
    
    return loss_cls, loss_reg

def weighted_KL_div(ps, qt, pos_w, neg_w):
    eps = 1e-10
    ps = ps + eps
    qt = qt + eps
    log_p = qt * torch.log(ps)
    log_p[:,0] *= neg_w
    log_p[:,1:] *= pos_w
    return -torch.mean(log_p)


# def bounded_regression_loss(Rs, Rt, gt, m, v=0.5):
#     loss = torch.sum(F.mse_loss(Rs, gt, reduction='none'), 1)
#     return torch.sum(loss * (loss + m > torch.sum(F.mse_loss(Rt, gt, reduction='none'), 1))) * v


def bounded_regression_loss(Rs, Rt, gt,gt_t, m, v=0.5):
    
    loss = F.mse_loss(Rs, gt)
    if loss + m > F.mse_loss(Rt, gt_t):
        return loss * v 
    else:
        loss.fill_(0)
        return loss
