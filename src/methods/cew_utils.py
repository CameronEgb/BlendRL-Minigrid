import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from scipy.spatial.distance import minkowski
import time
from kneed import KneeLocator
import igraph

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / (np.power(sigma, 2) + 1e-12)))

def R_regulator(sigma_1, sigma_2):
    return (1/2) * (sigma_1 + sigma_2)

def run_CLIP(X, mins, maxes, terms=None, eps=0.1, kappa=0.6, theta=1e-8):
    if terms is None: terms = []
    if not terms:
        for _ in range(X.shape[1]): terms.append([])
    
    for x in X:
        if not terms[0]:
            # First observation creates initial clusters
            for p in range(X.shape[1]):
                c_1p = x[p]
                # Sigma calculation matching original repo: R(left, right)
                left_width = np.sqrt(-1.0 * (np.power((mins[p] - x[p]) + theta, 2) / np.log(eps)))
                right_width = np.sqrt(-1.0 * (np.power((maxes[p] - x[p]) + theta, 2) / np.log(eps)))
                sigma_1p = R_regulator(left_width, right_width)
                terms[p].append({'center': c_1p, 'sigma': sigma_1p, 'support': 1})
            continue

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
                    # Fallback if somehow no neighbors found
                    new_sigma = (maxes[p] - mins[p]) / np.sqrt(-np.log(eps))
                elif jL_p is None:
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
    num_features = X.shape[1]
    sqrt_q = np.sqrt(num_features)
    
    if not Cs:
        max_clusters = len(X)
        centers = np.empty((max_clusters, num_features))
        radii = np.zeros(max_clusters)
        supports = np.zeros(max_clusters, dtype=int)
        
        centers[0] = X[0]
        radii[0] = 0.0
        supports[0] = 1
        num_clusters = 1
        start_idx = 1
    else:
        max_clusters = len(X) + len(Cs)
        centers = np.empty((max_clusters, num_features))
        radii = np.zeros(max_clusters)
        supports = np.zeros(max_clusters, dtype=int)
        
        for idx, C in enumerate(Cs):
            centers[idx] = C.center
            radii[idx] = C.radius
            supports[idx] = C.support
        num_clusters = len(Cs)
        start_idx = 0
        
    for i in range(start_idx, len(X)):
        x = X[i]
        
        # Sliced distance calculation (no list/array conversion inside the loop!)
        diff = centers[:num_clusters] - x
        D_i = np.linalg.norm(diff, axis=1) / sqrt_q
        
        inside_mask = D_i < radii[:num_clusters]
        if np.any(inside_mask):
            j = np.argmax(inside_mask)
            supports[j] += 1
            continue
            
        S_i = D_i + radii[:num_clusters]
        a = np.argmin(S_i)
        
        if S_i[a] > (2.0 * Dthr):
            centers[num_clusters] = x
            radii[num_clusters] = 0.0
            supports[num_clusters] = 1
            num_clusters += 1
        else:
            radii[a] = S_i[a] / 2.0
            supports[a] += 1
            n = supports[a]
            centers[a] = ((n - 1) * centers[a] + x) / n
            
    new_Cs = []
    for idx in range(num_clusters):
        C = Cluster(center=centers[idx], radius=radii[idx])
        C.support = supports[idx]
        new_Cs.append(C)
    return new_Cs

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
    def __init__(self, in_features, out_features, antecedents, rules, consequences=None, trainable_antecedents=False):
        super(FLC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if not antecedents:
            antecedents = [[] for _ in range(in_features)]
            
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
        
        self.input_terms = GaussianLayer(self.transformed_len, centers, sigmas, trainable=trainable_antecedents)
        
        if consequences is None:
            # Small random initialization instead of zeros to break symmetry
            self.consequences = Parameter(torch.randn(len(rules), out_features) * 0.01)
        else:
            self.consequences = Parameter(torch.tensor(consequences, dtype=torch.float32))
            
    @classmethod
    def from_shapes(cls, transformed_len, n_rules, out_features=1):
        flc = cls.__new__(cls)
        super(FLC, flc).__init__()
        flc.in_features = transformed_len
        flc.out_features = out_features
        flc.transformed_len = transformed_len
        flc.input_variable_ids = []
        flc.register_buffer('links', torch.zeros((transformed_len, n_rules), dtype=torch.float32))
        flc.register_buffer('links_mask', torch.zeros((transformed_len, n_rules), dtype=torch.float32))
        flc.register_buffer('feature_map', torch.zeros(transformed_len, dtype=torch.long))
        flc.input_terms = GaussianLayer(transformed_len, [0.0]*transformed_len, [1.0]*transformed_len)
        flc.consequences = Parameter(torch.zeros(n_rules, out_features, dtype=torch.float32))
        return flc

    def forward(self, X):
        if self.transformed_len == 0 or self.links.shape[1] == 0:
            return torch.zeros(X.shape[0], self.out_features, device=X.device)
        
        X_trans = X.index_select(1, self.feature_map)
        mems = self.input_terms(X_trans)
        
        # log(mems) is log-membership
        log_mems = torch.log(mems + 1e-12)
        
        # rules_log_act = sum_{terms in rule} log(mem_term)
        rules_log_act = torch.matmul(log_mems, self.links)
        
        # Back to linear space
        rules_act = torch.exp(rules_log_act)
        
        num = torch.matmul(rules_act, self.consequences)
        den = rules_act.sum(dim=1, keepdim=True)
        return num / torch.clamp(den, min=1e-12)

    def get_q_values(self, X):
        return self.forward(X)

    def get_rule_activations(self, X):
        X_trans = X.index_select(1, self.feature_map)
        mems = self.input_terms(X_trans)
        log_mems = torch.log(mems + 1e-12)
        rules_log_act = torch.matmul(log_mems, self.links)
        return torch.exp(rules_log_act)

class MultiFLC(nn.Module):
    def __init__(self, n_inputs, n_outputs, antecedents, rules, learning_rate=1e-3, cql_alpha=0.5):
        super(MultiFLC, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cql_alpha = cql_alpha
        self.antecedents = antecedents
        self.rules = rules
        
        if not antecedents:
            antecedents = [[] for _ in range(n_inputs)]
            
        # Original code used multiple MISO FLCs
        self.flcs = nn.ModuleList([FLC(n_inputs, 1, antecedents, rules) for _ in range(n_outputs)])
        self.learning_rate = learning_rate

    @classmethod
    def from_state_dict_shapes(cls, prefix: str, state_dict: dict, n_outputs: int = 2):
        mflc = cls.__new__(cls)
        super(MultiFLC, mflc).__init__()
        mflc.n_outputs = n_outputs
        mflc.cql_alpha = 1.0
        mflc.learning_rate = 1e-3
        mflc.antecedents = None
        mflc.rules = None
        flcs_list = []
        for out_idx in range(n_outputs):
            flc_prefix = f"{prefix}flcs.{out_idx}."
            links_key = f"{flc_prefix}links"
            cons_key = f"{flc_prefix}consequences"
            if links_key in state_dict and cons_key in state_dict:
                transformed_len = state_dict[links_key].shape[0]
                n_rules = state_dict[links_key].shape[1]
                out_features = state_dict[cons_key].shape[1]
                flcs_list.append(FLC.from_shapes(transformed_len, n_rules, out_features))
        mflc.flcs = nn.ModuleList(flcs_list)
        mflc.n_inputs = flcs_list[0].transformed_len if flcs_list else 0
        return mflc

    def forward(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        outputs = [flc(X_flat) for flc in self.flcs]
        return torch.cat(outputs, dim=1)

    def get_q_values(self, X):
        return self.forward(X)

    def get_action_and_value(self, obs, action=None):
        q = self.forward(obs)
        if action is None:
            action = torch.argmax(q, dim=1)
        log_probs = torch.log_softmax(q, dim=1)
        ent = -(torch.softmax(q, dim=1) * log_probs).sum(dim=1)
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1), ent, torch.max(q, dim=1)[0]

def run_FYD(rules, X, antecedents, top_k=None):
    if not rules or not antecedents: return rules, antecedents
    
    # 1. Calculate Scalar Cardinality for each term
    terms_sc = []
    for p in range(len(antecedents)):
        terms_sc.append([])
        for ant in antecedents[p]:
            sc = 0.0
            for x in X:
                sc += gaussian(x[p], ant['center'], ant['sigma'])
            terms_sc[p].append(sc)
    
    all_sc = [sc for p_sc in terms_sc for sc in p_sc]
    sc_min, sc_max = min(all_sc), max(all_sc)
    norm_sc = [(sc - sc_min) / (sc_max - sc_min + 1e-12) for sc in all_sc]
    
    # 2. Build Bipartite Graph for Usage and Closeness
    num_terms = sum(len(p_ants) for p_ants in antecedents)
    num_rules = len(rules)
    g = igraph.Graph()
    g.add_vertices(num_terms + num_rules)
    
    term_global_idx = 0
    term_map = [] # term_map[p][t_idx] = global_idx
    for p in range(len(antecedents)):
        term_map.append([])
        for t_idx in range(len(antecedents[p])):
            term_map[p].append(term_global_idx)
            term_global_idx += 1
            
    for r_idx, rule in enumerate(rules):
        for p_idx, t_idx in enumerate(rule['A']):
            g.add_edge(term_map[p_idx][t_idx], num_terms + r_idx)
            
    usage = np.array(g.degree(range(num_terms))) / num_rules
    closeness = np.array(g.closeness(range(num_terms)))
    closeness = np.nan_to_num(closeness, nan=0.0)
    
    usage_closeness = usage * closeness
    uc_min, uc_max = usage_closeness.min(), usage_closeness.max()
    norm_uc = (usage_closeness - uc_min) / (uc_max - uc_min + 1e-12)
    
    heuristic = np.array(norm_sc) * (1.0 - norm_uc)
    
    # 3. Kneedle for Cutoff
    valid_h = sorted([h for h in heuristic if h > 0])
    if not valid_h: return rules, antecedents
    
    if top_k is not None:
        cutoff = sorted(heuristic)[-top_k] if top_k < len(heuristic) else 0
    else:
        kneedle = KneeLocator(range(len(valid_h)), valid_h, curve="convex", direction="increasing")
        cutoff = kneedle.knee_y if kneedle.knee_y is not None else np.median(heuristic)
        
    # 4. Filter Terms
    term_global_idx = 0
    new_antecedents = [[] for _ in range(len(antecedents))]
    old_to_new_map = [{} for _ in range(len(antecedents))]
    
    for p in range(len(antecedents)):
        for t_idx in range(len(antecedents[p])):
            if heuristic[term_global_idx] >= cutoff:
                old_to_new_map[p][t_idx] = len(new_antecedents[p])
                new_antecedents[p].append(antecedents[p][t_idx])
            term_global_idx += 1
            
    # Ensure at least one term per feature
    for p in range(len(new_antecedents)):
        if not new_antecedents[p]:
            p_start = sum(len(antecedents[i]) for i in range(p))
            p_end = p_start + len(antecedents[p])
            best_t = np.argmax(heuristic[p_start:p_end])
            old_to_new_map[p][best_t] = 0
            new_antecedents[p].append(antecedents[p][best_t])
            
    # 5. Filter and Simplify Rules
    new_rules = []
    seen_A = set()
    for rule in rules:
        new_A = []
        keep_rule = True
        for p_idx, t_idx in enumerate(rule['A']):
            if t_idx in old_to_new_map[p_idx]:
                new_A.append(old_to_new_map[p_idx][t_idx])
            else:
                keep_rule = False
                break
        if keep_rule:
            A_tuple = tuple(new_A)
            if A_tuple not in seen_A:
                new_rules.append({'A': new_A, 'CF': rule['CF']})
                seen_A.add(A_tuple)
                
    if not new_rules:
        print("Warning: FYD pruning resulted in 0 rules. Falling back to original rules.")
        return rules, antecedents
        
    return new_rules, new_antecedents

class MamdaniAutoencoder(nn.Module):
    def __init__(self, in_features, antecedents, rules):
        super().__init__()
        # MUST allow training the antecedents
        self.flc = FLC(in_features, in_features, antecedents, rules, trainable_antecedents=True)
        
        consequences = np.zeros((len(rules), in_features))
        for r_idx, rule in enumerate(rules):
            for p_idx, t_idx in enumerate(rule['A']):
                consequences[r_idx, p_idx] = antecedents[p_idx][t_idx]['center']
        self.flc.consequences.data = torch.tensor(consequences, dtype=torch.float32)

    def forward(self, x):
        reconstruction = self.flc(x)
        activations = self.flc.get_rule_activations(x)
        return reconstruction, activations

def stabilize_antecedents(obs, antecedents, rules, device, lr=1e-3, epochs=10, batch_size=32):
    print(f"Stabilizing antecedents for {epochs} epochs...")
    in_features = obs.shape[1]
    model = MamdaniAutoencoder(in_features, antecedents, rules).to(device)
    optimizer = optim.Adam(model.flc.input_terms.parameters(), lr=lr)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _, target_activations = model(obs_tensor)
    
    for epoch in range(epochs):
        perm = torch.randperm(obs_tensor.size(0))
        for i in range(0, obs_tensor.size(0), batch_size):
            indices = perm[i:i + batch_size]
            batch_obs = obs_tensor[indices]
            batch_target_act = target_activations[indices]
            recon, act = model(batch_obs)
            loss = F.mse_loss(recon, batch_obs) + 0.1 * F.mse_loss(act, batch_target_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    trained_centers = model.flc.input_terms.centers.detach().cpu().numpy()
    trained_sigmas = model.flc.input_terms.sigmas.detach().cpu().numpy()
    
    term_idx = 0
    new_antecedents = [[] for _ in range(in_features)]
    for p in range(in_features):
        for ant in antecedents[p]:
            new_ant = ant.copy()
            new_ant['center'] = float(trained_centers[term_idx])
            new_ant['sigma'] = float(trained_sigmas[term_idx])
            new_antecedents[p].append(new_ant)
            term_idx += 1
            
    return new_antecedents
