import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from scipy.spatial.distance import minkowski
import time

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

def R_regulator(sigma_1, sigma_2):
    return (1/2) * (sigma_1 + sigma_2)

def run_CLIP(X, mins, maxes, terms=None, eps=0.2, kappa=0.6, theta=1e-8):
    if terms is None: terms = []
    if not terms:
        for _ in range(X.shape[1]): terms.append([])

    for x in X:
        if not terms[0]:
            for p in range(len(x)):
                c_1p = x[p]
                min_p = mins[p]
                max_p = maxes[p]
                left_width = np.sqrt(-1.0 * (np.power((min_p - x[p]) + theta, 2) / np.log(eps)))
                right_width = np.sqrt(-1.0 * (np.power((max_p - x[p]) + theta, 2) / np.log(eps)))
                sigma_1p = R_regulator(left_width, right_width)
                terms[p].append({'center': c_1p, 'sigma': sigma_1p, 'support': 1})
        else:
            for p in range(len(x)):
                SM_jps = [gaussian(x[p], A_jp['center'], A_jp['sigma']) for A_jp in terms[p]]
                if not SM_jps: continue
                j_star_p = np.argmax(SM_jps)
                
                if np.max(SM_jps) > kappa:
                    terms[p][j_star_p]['support'] += 1
                else:
                    # Find neighbors
                    jL_p = None
                    jR_p = None
                    jL_dist = float('inf')
                    jR_dist = float('inf')
                    
                    for j, A_jp in enumerate(terms[p]):
                        c_jp = A_jp['center']
                        dist = np.abs(c_jp - x[p])
                        if c_jp < x[p]:
                            if dist < jL_dist:
                                jL_dist = dist
                                jL_p = j
                        elif c_jp > x[p]:
                            if dist < jR_dist:
                                jR_dist = dist
                                jR_p = j
                    
                    new_c = x[p]
                    new_sigma = None
                    
                    if jL_p is None and jR_p is None:
                        continue # Should not happen with initial clusters
                    
                    if jL_p is None:
                        cR = terms[p][jR_p]['center']
                        sigma_R_old = terms[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR - x[p], 2) / np.log(eps)))
                        new_sigma = R_regulator(left_sigma_R, sigma_R_old)
                        terms[p][jR_p]['sigma'] = new_sigma
                    elif jR_p is None:
                        cL = terms[p][jL_p]['center']
                        sigma_L_old = terms[p][jL_p]['sigma']
                        right_sigma_L = np.sqrt(-1.0 * (np.power(cL - x[p], 2) / np.log(eps)))
                        new_sigma = R_regulator(right_sigma_L, sigma_L_old)
                        terms[p][jL_p]['sigma'] = new_sigma
                    else:
                        cR = terms[p][jR_p]['center']
                        sigma_R_old = terms[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR - x[p], 2) / np.log(eps)))
                        sigma_R = R_regulator(left_sigma_R, sigma_R_old)
                        
                        cL = terms[p][jL_p]['center']
                        sigma_L_old = terms[p][jL_p]['sigma']
                        right_sigma_L = np.sqrt(-1.0 * (np.power(cL - x[p], 2) / np.log(eps)))
                        sigma_L = R_regulator(right_sigma_L, sigma_L_old)
                        
                        new_sigma = R_regulator(sigma_R, sigma_L)
                        terms[p][jR_p]['sigma'] = terms[p][jL_p]['sigma'] = new_sigma
                    
                    terms[p].append({'center': new_c, 'sigma': new_sigma, 'support': 1})
    return terms

class Cluster:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.support = 1
    def add_support(self):
        self.support += 1

def general_euclidean_distance(x, y):
    q = len(x)
    return minkowski(x, y, p=2) / np.power(q, 0.5)

def run_ECM(X, Cs, Dthr):
    for x in X:
        if not Cs:
            Cs.append(Cluster(center=x, radius=0))
            continue
        
        D_i = [general_euclidean_distance(x, C.center) for C in Cs]
        
        match = False
        for j, C in enumerate(Cs):
            if D_i[j] < C.radius:
                C.add_support()
                match = True
                break
        if match: continue
        
        S_i = [D_i[j] + C.radius for j in range(len(Cs))]
        a = np.argmin(S_i)
        
        if S_i[a] > (2.0 * Dthr):
            Cs.append(Cluster(center=x, radius=0))
        else:
            Ca = Cs[a]
            Ca.radius = S_i[a] / 2.0
            Ca.add_support()
            n = Ca.support
            Ca.center = ((n - 1) * Ca.center + x) / n
    return Cs

def rule_creation(X, antecedents, consistency_check=True):
    rules = []
    weights = []
    for x in X:
        A_star_js = []
        CF = 1.0
        for p in range(len(x)):
            SM_jps = [gaussian(x[p], A_jp['center'], A_jp['sigma']) for A_jp in antecedents[p]]
            j_star_p = np.argmax(SM_jps)
            CF *= np.max(SM_jps)
            A_star_js.append(int(j_star_p))
        
        # Check uniqueness
        found = False
        for k, rule in enumerate(rules):
            if rule['A'] == A_star_js:
                weights[k] += 1.0
                rule['CF'] = min(rule['CF'], CF)
                found = True
                break
        if not found:
            rules.append({'A': A_star_js, 'CF': CF})
            weights.append(1.0)
            
    if consistency_check:
        # Simplified consistency check: keep the one with max weight for each unique antecedent
        unique_A = {}
        for r, w in zip(rules, weights):
            A_tuple = tuple(r['A'])
            if A_tuple not in unique_A or w > unique_A[A_tuple][1]:
                unique_A[A_tuple] = (r, w)
        rules = [val[0] for val in unique_A.values()]
        
    return antecedents, rules

class GaussianLayer(nn.Module):
    def __init__(self, in_features, centers, sigmas, trainable=True):
        super(GaussianLayer, self).__init__()
        self.centers = Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=trainable)
        self.sigmas = Parameter(torch.tensor(sigmas, dtype=torch.float32), requires_grad=trainable)
    def forward(self, x):
        return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / (torch.pow(self.sigmas, 2) + 1e-12)))

class FLC(nn.Module):
    def __init__(self, in_features, out_features, antecedents, rules, consequences=None):
        super(FLC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        unique_id = 0
        centers = []; sigmas = []; self.input_variable_ids = []
        for p in range(in_features):
            self.input_variable_ids.append([])
            for ant in antecedents[p]:
                centers.append(ant['center'])
                sigmas.append(ant['sigma'])
                ant['id'] = unique_id
                self.input_variable_ids[-1].append(unique_id)
                unique_id += 1
        self.transformed_len = unique_id
        
        links = np.zeros((self.transformed_len, len(rules)))
        for r_idx, rule in enumerate(rules):
            for p, t_idx in enumerate(rule['A']):
                links[antecedents[p][t_idx]['id'], r_idx] = 1
        self.register_buffer('links', torch.tensor(links, dtype=torch.float32))
        self.register_buffer('links_mask', (self.links == 0).float())
        
        self.input_terms = GaussianLayer(self.transformed_len, centers, sigmas)
        
        if consequences is None:
            self.consequences = Parameter(torch.zeros(len(rules), out_features))
        else:
            self.consequences = Parameter(torch.tensor(consequences, dtype=torch.float32))
            
        self.register_buffer('feature_map', torch.tensor([i for i, ids in enumerate(self.input_variable_ids) for _ in ids], dtype=torch.long))

    def forward(self, X):
        if self.transformed_len == 0 or self.links.shape[1] == 0:
            return torch.zeros(X.shape[0], self.out_features, device=X.device)
        
        # 1. Transform input
        X_trans = X.index_select(1, self.feature_map)
        
        # 2. Membership values
        mems = self.input_terms(X_trans)
        
        # 3. Rule applicability (Product T-norm)
        # mems: (B, T), links: (T, R), links_mask: (T, R)
        # Using log-sum-exp trick for stability or just product with mask
        # We'll use the original product logic
        rules_act = (mems.unsqueeze(2) * self.links.unsqueeze(0) + self.links_mask.unsqueeze(0)).prod(dim=1)
        
        # 4. Weighted Average Defuzzification
        # rules_act: (B, R), consequences: (R, O)
        num = torch.matmul(rules_act, self.consequences)
        den = rules_act.sum(dim=1, keepdim=True)
        return num / torch.clamp(den, min=1e-12)

class MultiFLC(nn.Module):
    def __init__(self, n_inputs, n_outputs, antecedents, rules, learning_rate=1e-3, cql_alpha=0.5):
        super(MultiFLC, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cql_alpha = cql_alpha
        
        # Original code used multiple MISO FLCs
        self.flcs = nn.ModuleList([FLC(n_inputs, 1, antecedents, rules) for _ in range(n_outputs)])
        self.learning_rate = learning_rate

    def forward(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        outputs = [flc(X_flat) for flc in self.flcs]
        return torch.cat(outputs, dim=1)

    def get_action_and_value(self, X, action=None):
        q = self.forward(X)
        if action is None:
            action = torch.argmax(q, dim=1)
        log_probs = torch.log_softmax(q, dim=1)
        ent = -(torch.softmax(q, dim=1) * log_probs).sum(dim=1)
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1), ent, torch.max(q, dim=1)[0]

def run_FYD(rules, X, antecedents, top_k=None):
    if not rules: return rules
    rule_supports = np.zeros(len(rules))
    for x in X:
        for r_idx, rule in enumerate(rules):
            cf = 1.0
            for p, t_idx in enumerate(rule['A']):
                cf *= gaussian(x[p], antecedents[p][t_idx]['center'], antecedents[p][t_idx]['sigma'])
            rule_supports[r_idx] += cf
    disc = np.zeros(len(rules))
    for i in range(len(rules)):
        sim = sum(sum(1 for a1, a2 in zip(rules[i]['A'], rules[j]['A']) if a1 == a2) / len(rules[i]['A']) 
                  for j in range(len(rules)) if i != j)
        disc[i] = 1.0 / (1.0 + sim)
    heuristic = rule_supports * disc
    indices = np.argsort(heuristic)[-(top_k if top_k else int(np.mean(heuristic > np.mean(heuristic)))): ]
    return [rules[i] for i in sorted(indices)]
