from sklearn.metrics import roc_curve,auc
import torch
from .measures import wrong_class,correct_class,MSP
import numpy as np

def accuracy(y_pred,y_true):
    '''Returns the accuracy in a batch'''
    return correct_class(y_pred,y_true).sum()/y_true.size(0)


def ROC_curve(loss, confidence, return_threholds = False):
    fpr, tpr, thresholds = roc_curve(loss.cpu(),(1-confidence).cpu())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr
    
def RC_curve(loss:torch.tensor, confidence:torch.tensor,
             coverages = None, return_thresholds:bool = False):
    loss = loss.view(-1)
    confidence = confidence.view(-1)
    n = len(loss)
    assert len(confidence) == n
    confidence,indices = confidence.sort(descending = True)
    loss = loss[indices]

    if coverages is not None:
        #deprecated
        coverages = torch.as_tensor(coverages,device = loss.device)
        thresholds = confidence.quantile(coverages)
        indices = torch.searchsorted(confidence,thresholds).minimum(torch.as_tensor(confidence.size(0)-1,device=loss.device))
    else:
        #indices = confidence.diff().nonzero().view(-1)
        indices = torch.arange(n,device=loss.device)
    coverages = (1 + indices)/n
    risks = (loss.cumsum(0)[indices])/n
    risks /= coverages
    coverages = np.r_[0.,coverages.cpu().numpy()]
    risks = np.r_[0.,risks.cpu().numpy()]

    if return_thresholds:
        thresholds = np.quantile(confidence.cpu().numpy(),1-coverages)
        return coverages, risks, thresholds
    else: return coverages, risks

def coverages_from_t(g:torch.tensor,t):
    return g.le(t.view(-1,1)).sum(-1)/g.size(0)


def SAC(risk:torch.tensor,confidence:torch.tensor,accuracy:float):
    coverages,risk = RC_curve(risk,confidence)
    target_risk = 1-accuracy
    coverages = coverages[risk<=target_risk]
    if coverages.size>0: return coverages[-1]
    else: return 0.0

def AUROC(loss,confidence):
    fpr,tpr = ROC_curve(loss,confidence)
    return auc(fpr, tpr)

def AURC(loss,confidence, coverages = None):
    coverages,risk_list = RC_curve(loss,confidence, coverages)
    return auc(coverages,risk_list)
def E_AURC(loss,confidence, coverages = None):
    return AURC(loss,confidence,coverages)-AURC(loss,1-loss,coverages)
def N_AURC(loss,confidence, coverages = None):
    return E_AURC(loss,confidence,coverages)/(loss.mean().item()-AURC(loss,1-loss,coverages))

def AUROC_fromlogits(y_pred,y_true,confidence = None, risk_fn = wrong_class):
    if confidence is None: confidence = MSP(y_pred)
    risk = risk_fn(y_pred,y_true).float()
    return AUROC(risk,confidence)

def AURC_fromlogits(y_pred,y_true,confidence = None, risk_fn = wrong_class, coverages = None):
    if confidence is None: confidence = MSP(y_pred)
    risk = risk_fn(y_pred,y_true).float()
    return AURC(risk,confidence,coverages)

class ReliabilityDiagram():
    def __init__(self, n_bins=10, bins_boundaries = (0,1)):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(*bins_boundaries, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    def ECE(self, risk:torch.tensor, confidences:torch.tensor):
        risk = 1-risk

        ece = torch.zeros(1, device=risk.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = risk[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.item()
    def __call__(self,risk:torch.tensor,confidences:torch.tensor):
        risk = 1-risk
        confs = []
        accs = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accs.append(risk[in_bin].float().mean().item())
                confs.append(confidences[in_bin].mean().item())
        return np.array(confs),np.array(accs)
    

def delta_ece(id:tuple,ood:tuple,n_bins = 15):
    ece = ReliabilityDiagram(n_bins).ECE
    return np.abs(ece(*id)-ece(*ood))

def delta_sac(id:tuple,ood:tuple,acc:float):
    return np.abs(SAC(*id,acc)-SAC(*ood,acc))

def rc(risk:torch.tensor,confidence:torch.tensor,coverage:float):
    risk = risk[confidence.sort(descending = True)[1]]
    n = int(coverage*risk.size(0))
    return risk[:n].mean()
def delta_rc(id:tuple,ood:tuple,c:float):
    return np.abs(rc(*id,c)-rc(*ood,c))

def r_gamma(risk:torch.tensor,confidences:torch.tensor,n_bins:int = 15,bins_boundaries:tuple = (0,1)):
    confs = []
    r = []
    bin_boundaries = torch.linspace(*bins_boundaries, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item())*confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            r.append(risk[in_bin].float().mean().item())
            confs.append(confidences[in_bin].mean().item())
    return np.array(confs),np.array(r)

def p_gamma(confidences:torch.tensor,n_bins:int = 15,bins_boundaries:tuple = (0,1)):
    confs = []
    p = []
    bin_boundaries = torch.linspace(*bins_boundaries, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item())*confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean().div(bin_upper-bin_lower)
        p.append(prop_in_bin)
        if prop_in_bin.item() > 0: confs.append(confidences[in_bin].mean().item())
        else: confs.append(np.nan)
    return np.array(confs),np.array(p)

def RC_pr(p:np.array,r:np.array):
    assert len(p) == len(r)
    coverages = np.flip(p).cumsum()
    return coverages,np.flip(r*p).cumsum()/coverages

def UQR(id:tuple,ood:tuple,n_bins = 15):
    RD = ReliabilityDiagram(n_bins)
    return np.abs(RD(*id)[1]-RD(*ood)[1]).mean()

def RCCR(p,r_i,r_o,coverages = np.linspace(0,1,20)):
    c_i,rc_i = RC_pr(p,r_i)
    c_o,rc_o = RC_pr(p,r_o)
    if coverages is None: coverages = np.concatenate((c_i,c_o)).sort()
    #implement other interpolation methods (non-linear)?
    rc_i = np.interp(coverages,c_i,rc_i)
    rc_o = np.interp(coverages,c_o,rc_o)
    return np.abs(rc_i-rc_o).sum()

