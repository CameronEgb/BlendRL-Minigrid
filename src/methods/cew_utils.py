import numpy as np
from scipy.spatial.distance import minkowski

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

def R_regulator(sigma_1, sigma_2):
    return (1/2) * (sigma_1 + sigma_2)

def run_CLIP(X, mins, maxes, terms=None, eps=0.2, kappa=0.6, theta=1e-8):
    if terms is None:
        terms = []
    
    # Initialize terms list for each feature
    if not terms:
        for _ in range(X.shape[1]):
            terms.append([])

    for x in X:
        if not terms[0]:
            # No fuzzy clusters yet, create the first fuzzy cluster for each feature
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
                SM_jps = []
                for j, A_jp in enumerate(terms[p]):
                    SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                    SM_jps.append(SM_jp)
                
                if not SM_jps:
                    continue

                j_star_p = np.argmax(SM_jps)

                if np.max(SM_jps) > kappa:
                    terms[p][j_star_p]['support'] += 1
                else:
                    jL_p = None
                    jR_p = None
                    jL_p_differences = []
                    jR_p_centers = []
                    jL_p_centers = []
                    
                    for j, A_jp in enumerate(terms[p]):
                        c_jp = A_jp['center']
                        if c_jp < x[p]:
                            jL_p_differences.append(np.abs(c_jp - x[p]))
                            jL_p_centers.append(j)
                        elif c_jp > x[p]:
                            # Will handle R neighbor next
                            pass
                    
                    if jL_p_differences:
                        jL_p = jL_p_centers[np.argmin(jL_p_differences)]

                    jR_p_differences = []
                    jR_p_centers = []
                    for j, A_jp in enumerate(terms[p]):
                        c_jp = A_jp['center']
                        if c_jp > x[p]:
                            jR_p_differences.append(np.abs(c_jp - x[p]))
                            jR_p_centers.append(j)
                    
                    if jR_p_differences:
                        jR_p = jR_p_centers[np.argmin(jR_p_differences)]

                    new_c = x[p]
                    new_sigma = None

                    if jL_p is None and jR_p is None:
                        # Should not happen given initial setup but for safety
                        continue

                    if jL_p is None:
                        cR_jp = terms[p][jR_p]['center']
                        sigma_R_jp = terms[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR_jp - x[p], 2) / np.log(eps)))
                        sigma_R = R_regulator(left_sigma_R, sigma_R_jp)
                        new_sigma = sigma_R
                        terms[p][jR_p]['sigma'] = new_sigma
                    elif jR_p is None:
                        cL_jp = terms[p][jL_p]['center']
                        sigma_L_jp = terms[p][jL_p]['sigma']
                        left_sigma_L = np.sqrt(-1.0 * (np.power(cL_jp - x[p], 2) / np.log(eps)))
                        sigma_L = R_regulator(left_sigma_L, sigma_L_jp)
                        new_sigma = sigma_L
                        terms[p][jL_p]['sigma'] = new_sigma
                    else:
                        cR_jp = terms[p][jR_p]['center']
                        sigma_R_jp = terms[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR_jp - x[p], 2) / np.log(eps)))
                        sigma_R = R_regulator(left_sigma_R, sigma_R_jp)

                        cL_jp = terms[p][jL_p]['center']
                        sigma_L_jp = terms[p][jL_p]['sigma']
                        left_sigma_L = np.sqrt(-1.0 * (np.power(cL_jp - x[p], 2) / np.log(eps)))
                        sigma_L = R_regulator(left_sigma_L, sigma_L_jp)

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
            C = Cluster(center=x, radius=0)
            Cs.append(C)
            continue
        
        D_i = []
        for j, C in enumerate(Cs):
            dist = general_euclidean_distance(x, C.center)
            D_i.append(dist)
            
        # Check if x belongs to any existing cluster
        found = False
        for j, C in enumerate(Cs):
            if D_i[j] <= C.radius:
                C.add_support()
                found = True
                break
        
        if found:
            continue
        
        # Find best cluster to update or create new
        S_i = []
        for j, C in enumerate(Cs):
            S_i.append(D_i[j] + C.radius)
        
        a = np.argmin(S_i)
        S_ia = S_i[a]
        
        if S_ia > (2.0 * Dthr):
            C = Cluster(center=x, radius=0)
            Cs.append(C)
        else:
            Ca = Cs[a]
            Ca.radius = S_ia / 2.0
            Ca.add_support()
            n = Ca.support
            Ca.center = ((n - 1) * Ca.center + x) / n
            
    return Cs

def rule_creation(X, antecedents):
    rules = []
    for x in X:
        CF = 1.0
        A_star_js = []
        for p in range(len(x)):
            SM_jps = []
            for j, A_jp in enumerate(antecedents[p]):
                SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                SM_jps.append(SM_jp)
            
            j_star_p = np.argmax(SM_jps)
            CF *= SM_jps[j_star_p]
            A_star_js.append(j_star_p)

        # Check for duplicates
        is_duplicate = False
        for r in rules:
            if r['A'] == A_star_js:
                is_duplicate = True
                break
        
        if not is_duplicate:
            rules.append({'A': A_star_js, 'CF': CF})
            
    return rules

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

class Gaussian(nn.Module):
    def __init__(self, in_features, centers=None, sigmas=None, trainable=True):
        super(Gaussian, self).__init__()
        self.in_features = in_features

        # initialize centers
        if centers is None:
            self.centers = Parameter(torch.randn(self.in_features))
        else:
            self.centers = Parameter(torch.tensor(centers, dtype=torch.float32))

        # initialize sigmas
        if sigmas is None:
            self.sigmas = Parameter(torch.abs(torch.randn(self.in_features)))
        else:
            self.sigmas = Parameter(torch.abs(torch.tensor(sigmas, dtype=torch.float32)))

        self.centers.requires_grad = trainable
        self.sigmas.requires_grad = trainable

    def forward(self, x):
        return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.sigmas, 2)))

class FLC(nn.Module):
    def __init__(self, in_features, out_features, antecedents, rules, consequences=None):
        super(FLC, self).__init__()
        self.in_features = in_features

        num_of_antecedents = np.zeros(in_features).astype('int32')
        unique_id = 0
        gaussians = {'centers': [], 'sigmas': []}
        self.input_variable_ids = []
        self.transformed_x_length = 0
        for input_variable_idx in range(in_features):
            num_of_antecedents[input_variable_idx] = len(antecedents[input_variable_idx])
            self.input_variable_ids.append(set())
            for term_idx, antecedent in enumerate(antecedents[input_variable_idx]):
                gaussians['centers'].append(antecedent['center'])
                gaussians['sigmas'].append(antecedent['sigma'])
                antecedent['id'] = unique_id
                self.input_variable_ids[-1].add(unique_id)
                unique_id += 1
        self.transformed_x_length = unique_id

        n_rules = len(rules)
        links = np.zeros((self.transformed_x_length, n_rules))

        for rule_idx, rule in enumerate(rules):
            for input_variable_idx, term_idx in enumerate(rule['A']):
                new_term_idx = antecedents[input_variable_idx][term_idx]['id']
                links[new_term_idx, rule_idx] = 1

        self.register_buffer('links_between_antecedents_and_rules', torch.tensor(links, dtype=torch.float32))
        self.input_terms = Gaussian(in_features=self.transformed_x_length, centers=gaussians['centers'],
                                    sigmas=gaussians['sigmas'], trainable=False)

        if consequences is None:
            self.consequences = Parameter(torch.randn(n_rules, out_features) * 0.01)
        else:
            self.consequences = Parameter(torch.tensor(consequences, dtype=torch.float32))

    def __transform(self, X):
        batch_size = X.shape[0]
        new_X = torch.zeros((batch_size, self.transformed_x_length), device=X.device)
        for input_variable_idx, indices_to_repeat_for in enumerate(self.input_variable_ids):
            min_idx = min(indices_to_repeat_for)
            max_idx = max(indices_to_repeat_for) + 1
            copies = len(indices_to_repeat_for)
            new_X[:, min_idx:max_idx] = X[:, input_variable_idx].unsqueeze(1).repeat(1, copies)
        return new_X

    def forward(self, X):
        X_transformed = self.__transform(X)
        antecedents_memberships = self.input_terms(X_transformed)
        # shape: (batch, terms, rules)
        terms_to_rules = antecedents_memberships.unsqueeze(2) * self.links_between_antecedents_and_rules
        
        # We need to handle the product only for active links
        # A trick: set inactive links to 1.0 so they don't affect the product
        mask = (self.links_between_antecedents_and_rules == 0)
        terms_to_rules = terms_to_rules + mask.unsqueeze(0).float()
        
        rules_applicability = terms_to_rules.prod(dim=1)
        
        # MISO: self.consequences is (n_rules, 1)
        numerator = (rules_applicability * self.consequences.squeeze(1)).sum(dim=1)
        denominator = rules_applicability.sum(dim=1)
        denominator = torch.clamp(denominator, min=1e-12)
        return numerator / denominator

class MultiFLC(nn.Module):
    def __init__(self, n_inputs, n_outputs, antecedents, rules, learning_rate=3e-4, cql_alpha=0.5):
        super(MultiFLC, self).__init__()
        self.flcs = nn.ModuleList([FLC(n_inputs, 1, antecedents, rules) for _ in range(n_outputs)])
        self.learning_rate = learning_rate
        self.cql_alpha = cql_alpha
        self.n_outputs = n_outputs

    def forward(self, X):
        outputs = [flc(X) for flc in self.flcs]
        return torch.stack(outputs, dim=1)

    def get_action_probs(self, X):
        """Returns a probability distribution over actions."""
        q_values = self.forward(X)
        return torch.softmax(q_values, dim=1)

    def get_action_and_value(self, X, action=None):
        q_values = self.forward(X)
        if action is None:
            action = torch.argmax(q_values, dim=1)
        
        # We need to return (action, logprob, entropy, value) for the evaluator
        # But this is a Q-learning agent, not PPO.
        # Let's return what IQLAgent evaluator expects if possible.
        # Looking at EnvironmentEvaluatorCallback, it calls get_action_and_value
        # and expects (action, logprob, entropy, value).
        
        log_probs = torch.log_softmax(q_values, dim=1)
        action_logprob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Entropy for a deterministic policy is 0, but we can use softmax entropy
        probs = torch.softmax(q_values, dim=1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
        
        value = torch.max(q_values, dim=1)[0]
        
        return action, action_logprob, entropy, value

def run_FYD(rules, X, antecedents, top_k=None):
    """
    Frequent-Yet-Discernible simplification.
    1. Calculate support (frequency) for each rule.
    2. Calculate discernibility (how unique/important the rule is).
    3. Prune rules.
    """
    if not rules:
        return rules
        
    rule_supports = np.zeros(len(rules))
    # Calculate membership of each X in each rule
    for i, x in enumerate(X):
        memberships = []
        for r_idx, rule in enumerate(rules):
            CF = 1.0
            for p, term_idx in enumerate(rule['A']):
                CF *= gaussian(x[p], antecedents[p][term_idx]['center'], antecedents[p][term_idx]['sigma'])
            memberships.append(CF)
        
        # Support is cumulative membership
        rule_supports += np.array(memberships)
        
    # Discernibility: how much does this rule stand out?
    # One simple measure: 1 / (similarity to other rules)
    discernibility = np.zeros(len(rules))
    for i in range(len(rules)):
        sim_sum = 0
        for j in range(len(rules)):
            if i == j: continue
            # Count matching antecedents
            matches = sum(1 for a1, a2 in zip(rules[i]['A'], rules[j]['A']) if a1 == a2)
            sim_sum += matches / len(rules[i]['A'])
        discernibility[i] = 1.0 / (1.0 + sim_sum)
        
    # Heuristic: support * discernibility
    heuristic = rule_supports * discernibility
    
    if top_k is None:
        # Keep rules above average heuristic?
        threshold = np.mean(heuristic)
        indices = np.where(heuristic >= threshold)[0]
    else:
        indices = np.argsort(heuristic)[-top_k:]
        
    new_rules = [rules[i] for i in sorted(indices)]
    print(f"FYD: Pruned {len(rules)} rules to {len(new_rules)}")
    return new_rules
