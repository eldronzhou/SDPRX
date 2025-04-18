import numpy as np
from scipy import stats, special, linalg
from collections import Counter
from joblib import Parallel, delayed

def initial_state(data1, data2, M, N1, N2, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, alpha=1.0, a0k=.5, b0k=.5):
    num_clusters = sum(M) 
    cluster_ids = range(num_clusters)

    state = {
        'cluster_ids_': cluster_ids, 
        'beta_margin1_': data1,
        'beta_margin2_': data2,
        'b1': np.zeros(len(data1)), 
        'b2': np.zeros(len(data2)), 
        'N1_': N1,
        'N2_': N2,
        'beta1': np.zeros(len(data1)),
        'beta2': np.zeros(len(data2)),
        'num_clusters_': num_clusters,
        'alpha': np.array([alpha]*4),
        'hyperparameters_': {
            "a0k": a0k,
            "b0k": b0k,
            "a0": 0.1,
            "b0": 0.1,
        },
        'suffstats': np.array([0]*(num_clusters)),
        'assignment1': np.random.randint(num_clusters, size=len(data1)),
	'assignment2': np.array([0]*len(data2)),
        'population': np.array([0]*4),
        'pi': np.array([alpha / num_clusters]*num_clusters),
        'pi_pop': np.array([.25, .25, .25, .25]),
        'pi_cluster': [np.array(1.0/M[i]) for i in range(len(M))],
        'V': [np.array([0]*M[i]) for i in range(len(M))],
        'cluster_var': np.array([0]*(num_clusters)),
        'varg1': np.array([0.0]*len(ld_boundaries1)),
        'varg2': np.array([0.0]*len(ld_boundaries1)),
        'h2_1': 0,
        'h2_2': 0,
        'eta': 1
    }
   
    # define indexes 
    state['population'][0] = 1 # null
    state['population'][1] = M[1] + 1 # pop 1 specific
    state['population'][2] = M[1] + M[2] + 1 # pop 2 specific
    state['population'][3] = num_clusters # shared with correlation
    
    tmp1 = []; tmp2 = []
    for j in range(len(ld_boundaries1)):
        start_i1 = ld_boundaries1[j][0]
        end_i1 = ld_boundaries1[j][1]
        start_i2 = ld_boundaries2[j][0]
        end_i2 = ld_boundaries2[j][1]
        tmp1.append(np.setdiff1d(np.array(range(end_i1-start_i1)), idx1_shared[j])+start_i1)
        tmp2.append(np.setdiff1d(np.array(range(end_i2-start_i2)), idx2_shared[j])+start_i2)
        state['assignment2'][idx2_shared[j]+start_i2] = state['assignment1'][idx1_shared[j]+start_i1]

    state['idx_pop1'] = np.concatenate(tmp1)
    state['idx_pop2'] = np.concatenate(tmp2)

    state['assignment1'][state['idx_pop1']] = np.random.randint(M[1], size=len(state['idx_pop1']))
    state['assignment2'][state['idx_pop2']] = np.random.randint(low=M[1]+1, high=M[1]+M[2], size=len(state['idx_pop2']))
    
    return state


def calc_b(j, state, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    ref_ld1 = ref_ld_mat1[j]
    ref_ld2 = ref_ld_mat2[j]
    shrink_ld1 = ref_ld1; shrink_ld2 = ref_ld2
    b1 = state['eta']*np.dot(state['A1'][j], state['beta_margin1_'][start_i1:end_i1]) - state['eta']**2 * \
    (np.dot(state['B1'][j], state['beta1'][start_i1:end_i1]) - np.diag(state['B1'][j])*state['beta1'][start_i1:end_i1])
    b2 = state['eta']*np.dot(state['A2'][j], state['beta_margin2_'][start_i2:end_i2]) - state['eta']**2 * \
    (np.dot(state['B2'][j], state['beta2'][start_i2:end_i2]) - np.diag(state['B2'][j])*state['beta2'][start_i2:end_i2])
    state['b1'][start_i1:end_i1] = b1
    state['b2'][start_i2:end_i2] = b2


def vectorized_random_choice(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    k[np.where(k == len(items))] = len(items) - 1
    return items[k]

def sample_assignment(j, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2, state, VS, rho):
    start_i1 = ld_boundaries1[j][0]; end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]; end_i2 = ld_boundaries2[j][1]
    m = state['num_clusters_']; N1 = state['N1_']; N2 = state['N2_']
    b1 = state['b1'][start_i1:end_i1].reshape((end_i1-start_i1, 1))
    b2 = state['b2'][start_i2:end_i2].reshape((end_i2-start_i2, 1))
    B1 = state['B1'][j]; B2 = state['B2'][j]

    log_prob_mat = np.zeros((len(idx1_shared[j]), m))
    assignment1 = np.zeros(len(b1))
    assignment2 = np.zeros(len(b2))
    
    # null or population 1 specific
    idx = range(0, state['population'][1])
    cluster_var = state['cluster_var'][idx]
    pi = np.array(state['pi'])[idx]
    C = -.5 * np.log(state['eta']**2*N1*np.outer(np.diag(B1[idx1_shared[j],:][:,idx1_shared[j]]), cluster_var) + 1) + \
        np.log( pi + 1e-40 )
    a = (N1*b1[idx1_shared[j]])**2 / (2 * np.add.outer(state['eta']**2 * N1 * np.diag(B1[idx1_shared[j],:][:,idx1_shared[j]]),  1.0/cluster_var[1:]) )
    log_prob_mat[:,idx] = np.insert(a, 0, 0, axis=1) + C
    
    # population 2 specific
    idx = range(state['population'][1], state['population'][2])
    cluster_var = state['cluster_var'][idx]
    pi = np.array(state['pi'])[idx]
    C = -.5 * np.log(state['eta']**2*N2*np.outer(np.diag(B2[idx2_shared[j],:][:,idx2_shared[j]]), cluster_var) + 1) + \
        np.log( pi + 1e-40 )
    a = (N2*b2[idx2_shared[j]])**2 / (2 * np.add.outer(state['eta']**2 * N2 * np.diag(B2[idx2_shared[j],:][:,idx2_shared[j]]),  1.0/cluster_var) )
    log_prob_mat[:,idx] = a + C
    
    # shared with correlation
    idx = range(state['population'][2], state['population'][3])
    cluster_var = state['cluster_var'][idx]
    pi = np.array(state['pi'])[idx]
    ak1 = np.add.outer(.5*N1*state['eta']**2*np.diag(B1[idx1_shared[j],:][:,idx1_shared[j]]), .5/((1-rho**2)*cluster_var))
    ak2 = np.add.outer(.5*N2*state['eta']**2*np.diag(B2[idx2_shared[j],:][:,idx2_shared[j]]), .5/((1-rho**2)*cluster_var))
    ck = rho / ((1-rho**2)*cluster_var)
    mu1 = (2*b1[idx1_shared[j]] + (N2*1.0/N1)*b2[idx2_shared[j]]*(ck/ak2)) / (4*ak1/N1 - ck**2/(ak2*N1))
    mu2 = (2*b2[idx2_shared[j]] + (N1*1.0/N2)*b1[idx1_shared[j]]*(ck/ak1)) / (4*ak2/N2 - ck**2/(ak1*N2))
        
    C = -.5*np.log(4*ak1*ak2-ck**2) - .5*np.log(1-rho**2) - np.log(cluster_var) + np.log( pi + 1e-40 )
        
    a = ak1*mu1*mu1 + ak2*mu2*mu2 - ck*mu1*mu2
    log_prob_mat[:,idx] = a + C
    
    logexpsum = special.logsumexp(log_prob_mat, axis=1).reshape((len(idx1_shared[j]), 1))
    prob_mat = np.exp(log_prob_mat - logexpsum)
    
    assignment_shared = vectorized_random_choice(prob_mat.T, np.array(state['cluster_ids_']))
    assignment1[idx1_shared[j]] = assignment_shared
    assignment2[idx2_shared[j]] = assignment_shared

    # pop1 specific variants
    idx_pop1 = np.setdiff1d(np.array(range(len(b1))), idx1_shared[j]) 
    if (len(idx_pop1) > 0):
    	idx = range(0, state['population'][1])
    	cluster_var = state['cluster_var'][idx]
    	pi = np.array(state['pi'])[idx]
    	C = -.5 * np.log(state['eta']**2*N1*np.outer(np.diag(B1[idx_pop1,:][:,idx_pop1]), cluster_var) + 1) + \
    		np.log( pi + 1e-40 )
    	a = (N1*b1[idx_pop1])**2 / (2 * np.add.outer(state['eta']**2 * N1 * np.diag(B1[idx_pop1,:][:,idx_pop1]),  1.0/cluster_var[1:]) )
    	log_prob_mat = np.insert(a, 0, 0, axis=1) + C
    	logexpsum = special.logsumexp(log_prob_mat, axis=1).reshape((len(idx_pop1), 1))
    	prob_mat = np.exp(log_prob_mat - logexpsum)
    	assignment1[idx_pop1] = vectorized_random_choice(prob_mat.T, np.array(state['cluster_ids_'])[idx])

    # pop2 specific variants
    idx_pop2 = np.setdiff1d(np.array(range(len(b2))), idx2_shared[j]) 
    if (len(idx_pop2) > 0):
    	idx = list(range(1)) + list(range(state['population'][1], state['population'][2]))
    	cluster_var = state['cluster_var'][idx]
    	pi = np.array(state['pi'])[idx]
    	C = -.5 * np.log(state['eta']**2*N2*np.outer(np.diag(B2[idx_pop2,:][:,idx_pop2]), cluster_var) + 1) + \
    		np.log( pi + 1e-40 )
    	a = (N2*b2[idx_pop2])**2 / (2 * np.add.outer(state['eta']**2 * N2 * np.diag(B2[idx_pop2,:][:,idx_pop2]),  1.0/cluster_var[1:]) )
    	log_prob_mat = np.insert(a, 0, 0, axis=1) + C
    	logexpsum = special.logsumexp(log_prob_mat, axis=1).reshape((len(idx_pop2), 1))
    	prob_mat = np.exp(log_prob_mat - logexpsum)
    	assignment2[idx_pop2] = vectorized_random_choice(prob_mat.T, np.array(state['cluster_ids_'])[idx])

    return assignment1, assignment2


def sample_beta(j, state, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2, rho, VS=True):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    N1 = state['N1_']; N2 = state['N2_']
    beta_margin1 = state['beta_margin1_'][start_i1:end_i1]; beta_margin2 = state['beta_margin2_'][start_i2:end_i2]
    A1 = state['A1'][j]; B1 = state['B1'][j]
    A2 = state['A2'][j]; B2 = state['B2'][j]
    cluster_var1 = state['cluster_var'][state['assignment1'][start_i1:end_i1]]
    cluster_var2 = state['cluster_var'][state['assignment2'][start_i2:end_i2]]
    
    beta1 = np.zeros(len(beta_margin1)); beta2 = np.zeros(len(beta_margin2))
    
    # null
    idx1_null = state['assignment1'][start_i1:end_i1] == 0
    idx2_null = state['assignment2'][start_i2:end_i2] == 0
     
    # pop1 specific
    # only considering beta1 is sufficient
    idx1 = (state['assignment1'][start_i1:end_i1] >= 1) & (state['assignment1'][start_i1:end_i1] < state['population'][1])
    
    # pop2 speicifc 
    idx2 = (state['assignment2'][start_i2:end_i2] >= state['population'][1]) \
        & (state['assignment2'][start_i2:end_i2] < state['population'][2])
    
    # shared with correlation
    idx3_1 = (state['assignment1'][start_i1:end_i1] >= state['population'][2]) \
        & (state['assignment1'][start_i1:end_i1] < state['population'][3])
    idx3_2 = (state['assignment2'][start_i2:end_i2] >= state['population'][2]) \
	    & (state['assignment2'][start_i2:end_i2] < state['population'][3])
    
    idx_pop1 = np.logical_or(idx1, idx3_1)
    idx_pop2 = np.logical_or(idx2, idx3_2)

    if all(idx1_null) and all(idx2_null):
        # all SNPs in this block are non-causal
        pass
    elif sum(idx1) > 1 and sum(idx_pop2) == 0:
    	shrink_ld = B1[idx1,:][:,idx1]
    	mat = state['eta']**2*N1*shrink_ld + np.diag(1.0 / cluster_var1[idx1])
    	chol, low = linalg.cho_factor(mat, overwrite_a=False)
    	cov_mat = linalg.cho_solve((chol, low), np.eye(chol.shape[0])) 
    	mu = state['eta']*N1*np.dot(cov_mat, A1[:, idx1].T).dot(beta_margin1)
    	beta1[idx1 == 1] = sample_MVN(mu, cov_mat)
    elif sum(idx1) == 1 and sum(idx_pop2) == 0:
        var_k = cluster_var1[idx1]
        const = var_k / (var_k*state['eta']**2*np.squeeze(B1[idx1,:][:,idx1]) + 1.0/N1)
        bj = state['b1'][start_i1:end_i1][idx1]
        beta1[idx1 == 1] = np.sqrt(const*1.0/N1)*stats.norm.rvs() + const*bj
    elif sum(idx2) > 1 and sum(idx_pop1) == 0:
    	shrink_ld = B2[idx2,:][:,idx2]
    	mat = state['eta']**2*N2*shrink_ld + np.diag(1.0 / cluster_var2[idx2])
    	chol, low = linalg.cho_factor(mat, overwrite_a=False)
    	cov_mat = linalg.cho_solve((chol, low), np.eye(chol.shape[0]))
    	mu = state['eta']*N2*np.dot(cov_mat, A2[:, idx2].T).dot(beta_margin2)
    	beta2[idx2 == 1] = sample_MVN(mu, cov_mat)
    elif sum(idx2) == 1 and sum(idx_pop1) == 0:
        var_k = cluster_var2[idx2]
        const = var_k / (var_k*state['eta']**2*np.squeeze(B2[idx2,:][:,idx2]) + 1.0/N2)
        bj = state['b2'][start_i2:end_i2][idx2]
        beta2[idx2 == 1] = np.sqrt(const*1.0/N2)*stats.norm.rvs() + const*bj
    else:
        # two population LD matrix
        shrink_ld = np.block([[N1*B1[idx_pop1,:][:,idx_pop1], np.zeros((sum(idx_pop1), sum(idx_pop2)))],
             [np.zeros((sum(idx_pop2), sum(idx_pop1))), N2*B2[idx_pop2,:][:,idx_pop2]]])
        
        # variance covariance matrix for beta
        idx_cor1 = np.where(state['assignment1'][start_i1:end_i1][idx_pop1] >= state['population'][2])[0]
        idx_cor2 = np.where(state['assignment2'][start_i2:end_i2][idx_pop2] >= state['population'][2])[0]

        diag1 = np.diag(1.0/cluster_var1[idx_pop1])
        cor1 = np.zeros((sum(idx_pop1), sum(idx_pop2)))
        diag2 = np.diag(1.0/cluster_var2[idx_pop2])
        cor2 = np.zeros((sum(idx_pop2), sum(idx_pop1)))
        
        for i in range(len(idx_cor1)):
            cor1[idx_cor1[i],idx_cor2[i]] = -rho/(1-rho**2)*diag1[idx_cor1[i],idx_cor1[i]]
            cor2[idx_cor2[i],idx_cor1[i]] = -rho/(1-rho**2)*diag1[idx_cor1[i],idx_cor1[i]]
            diag1[idx_cor1[i],idx_cor1[i]] = 1.0/(1-rho**2)*diag1[idx_cor1[i],idx_cor1[i]]
            diag2[idx_cor2[i],idx_cor2[i]] = 1.0/(1-rho**2)*diag2[idx_cor2[i],idx_cor2[i]]
        
        var_mat = np.block([[diag1, cor1],
                    [cor2, diag2]])
        
        mat = state['eta']**2*shrink_ld + var_mat
        
        chol, low = linalg.cho_factor(mat, overwrite_a=False)
        cov_mat = linalg.cho_solve((chol, low), np.eye(chol.shape[0])) 
        
        # A matrix
        A_gamma = np.concatenate([N1*np.dot(A1[idx_pop1,:], state['beta_margin1_'][start_i1:end_i1]), 
               N2*np.dot(A2[idx_pop2,:], state['beta_margin2_'][start_i2:end_i2])])
        
        mu = state['eta']*np.dot(cov_mat, A_gamma)
        beta_tmp = sample_MVN(mu, cov_mat)
        beta1[idx_pop1] = beta_tmp[0:sum(idx_pop1)]
        beta2[idx_pop2] = beta_tmp[sum(idx_pop1):]
        
    state['beta1'][start_i1:end_i1] = beta1
    state['beta2'][start_i2:end_i2] = beta2


def sample_MVN(mu, cov):
    rv = stats.norm.rvs(size=mu.shape[0])
    C = linalg.cholesky(cov, lower=True)
    return np.dot(C, rv) + mu


def compute_varg(j, state, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    ref_ld1 = ref_ld_mat1[j]
    ref_ld2 = ref_ld_mat2[j]
    state['varg1'][j] = np.sum(state['beta1'][start_i1:end_i1] * np.dot(ref_ld1, state['beta1'][start_i1:end_i1]))
    state['varg2'][j] = np.sum(state['beta2'][start_i2:end_i2] * np.dot(ref_ld2, state['beta2'][start_i2:end_i2]))


def calc_num(j, state, ld_boundaries1, ld_boundaries2):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    A1 = state['A1'][j]
    A2 = state['A2'][j]
    return state['N1_']*np.dot(state['beta_margin1_'][start_i1:end_i1], np.dot(A1, (state['beta1'][start_i1:end_i1]))) + \
        state['N2_']*np.dot(state['beta_margin2_'][start_i2:end_i2], np.dot(A2, (state['beta2'][start_i2:end_i2])))

def calc_denum(j, state, ld_boundaries1, ld_boundaries2):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    B1 = state['B1'][j]
    B2 = state['B2'][j]
    beta1 = state['beta1'][start_i1:end_i1]
    beta2 = state['beta2'][start_i2:end_i2]
    return state['N1_']*np.dot(beta1, np.dot(B1, beta1)) + state['N2_']*np.dot(beta2, np.dot(B2, beta2))

def sample_eta(state, ld_boundaries1, ld_boundaries2):
    num = np.sum([calc_num(j, state, ld_boundaries1, ld_boundaries2) for j in range(len(ld_boundaries1))])
    denum = np.sum([calc_denum(j, state, ld_boundaries1, ld_boundaries2) for j in range(len(ld_boundaries1))])
    mu = num / (denum + 1e-6)
    var = 1.0 / (denum+1e-6)
    return np.sqrt(var)*stats.norm.rvs() + mu


def sample_sigma2(state, rho, VS=True):
    b = np.zeros(state['num_clusters_'])
    a = np.array(list(state['suffstats'].values()) ) / 2.0 + state['hyperparameters_']['a0k'] 
    
    table1 = [[] for i in range(state['num_clusters_'])]
    for i in range(len(state['assignment1'])):
        table1[state['assignment1'][i]].append(i)

    table2 = [[] for i in range(state['num_clusters_'])]
    for i in range(len(state['assignment2'])):
        table2[state['assignment2'][i]].append(i)
    
    # pop1 specifc
    for i in range(1, state['population'][1]):
        b[i] = np.sum(state['beta1'][table1[i]]**2) / 2.0 + state['hyperparameters_']['b0k']
    
    # pop2 specifc
    for i in range(state['population'][1], state['population'][2]):
        b[i] = np.sum(state['beta2'][table2[i]]**2) / 2.0 + state['hyperparameters_']['b0k']
    
    # shared with correlation
    for i in range(state['population'][2], state['population'][3]):
        a[i] += state['suffstats'][i] / 2.0
        beta1 = state['beta1'][table1[i]]
        beta2 = state['beta2'][table2[i]]
        b[i] = np.sum( (beta1**2 + beta2**2 - 2*rho*beta1*beta2) / (2*(1-rho**2)) ) + state['hyperparameters_']['b0k']
    
    out = np.array([0.0]*state['num_clusters_'])
    if VS is True:
        out[1:] = stats.invgamma(a=a[1:], scale=b[1:]).rvs()
        out[0] = 0
    else: 
        out = dict(zip(range(0, state['num_clusters_']), stats.invgamma(a=a, scale=b).rvs()))
    return out


def update_suffstats(state):
    assn = np.concatenate([state['assignment1'], state['assignment2'][state['idx_pop2']]])
    suff_stats = dict(Counter(assn))
    suff_stats.update(dict.fromkeys(np.setdiff1d(range(state['num_clusters_']), list(suff_stats.keys())), 0))
    suff_stats = {k:suff_stats[k] for k in sorted(suff_stats)}
    return suff_stats

def sample_V(state):
    for j in range(1,4):
        m = len(state['V'][j])
        suffstats = np.array(list(state['suffstats'].values())[state['population'][j-1]:state['population'][j]])
        a = 1 + suffstats[:-1]
        b = state['alpha'][j] + np.cumsum(suffstats[::-1])[:-1][::-1]
        sample_val = stats.beta(a=a, b=b).rvs()
        if 1 in sample_val:
            idx = np.argmax(sample_val == 1)
            sample_val[idx+1:] = 0
            sample_return = dict(zip(range(m-1), sample_val))
            sample_return[m-1] = 0
        else:
            sample_return = dict(zip(range(m-1), sample_val))
            sample_return[m-1] = 1
        state['V'][j] = list(sample_return.values())

def sample_pi_pop(state):
    m = np.array([0.0]*4)
    null_pop1 = np.sum(state['assignment1'][state['idx_pop1']] == 0)
    null_pop2 = np.sum(state['assignment2'][state['idx_pop2']] == 0)
    nonnull_pop1 = len(state['idx_pop1']) - null_pop1
    nonnull_pop2 = len(state['idx_pop2']) - null_pop2
    m[0] += state['suffstats'][0] - null_pop1- null_pop2
    m[1] += np.sum(list(state['suffstats'].values())[1:state['population'][1]]) - nonnull_pop1
    m[2] += np.sum(list(state['suffstats'].values())[state['population'][1]:state['population'][2]]) - nonnull_pop2
    m[3] += np.sum(list(state['suffstats'].values())[state['population'][2]:state['population'][3]])
    state['suff_pop'] = m
    state['pi_pop'] = dict(zip(range(0, 4), stats.dirichlet(m+1).rvs()[0]))
        
# Compute pi
def update_p(state):
    state['pi'][0] = state['pi_pop'][0]
    for j in range(1,4):
        m = len(state['V'][j])
        V = state['V'][j]
        a = np.cumprod(1-np.array(V)[0:(m-2)])*V[1:(m-1)]
        pi = dict()
        pi[0] = state['V'][j][0]
        pi.update(dict(zip(range(1, m), a)))   
        pi[m-1] = 1 - np.sum(list(pi.values())[0:(m-1)])

        # last p may be less than 0 due to rounding error
        if pi[m-1] < 0: 
            pi[m-1] = 0
        state['pi_cluster'][j] = list(pi.values())
        idx = range(state['population'][j-1], state['population'][j])
        state['pi'][idx] = np.array(state['pi_cluster'][j])*state['pi_pop'][j]

# Sample alpha
def sample_alpha(state):
    for j in range(1,4):
        m = np.size(np.where( np.array(state['V'][j]) != 0)); V = state['V'][j]
        a = state['hyperparameters_']['a0'] + m - 1
        b = state['hyperparameters_']['b0'] - np.sum( np.log( 1 - np.array(V[0:(m-1)]) ) )
        state['alpha'][j] = stats.gamma(a=a, scale=1.0/b).rvs()


def gibbs_stick_break(state, rho, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2, n_threads, force_shared, VS=True):
    state['cluster_var'] = sample_sigma2(state, rho, VS)

    for j in range(len(ld_boundaries1)):
        calc_b(j, state, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2)
        
    tmp = Parallel(n_jobs=n_threads)(delayed(sample_assignment)(j=j, idx1_shared=idx1_shared, idx2_shared=idx2_shared, ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, ref_ld_mat1=ref_ld_mat1, ref_ld_mat2=ref_ld_mat2, state=state, rho=rho, VS=True) for j in range(len(ld_boundaries1)))
    state['assignment1'] = np.concatenate([tmp[j][0].astype(int) for j in range(len(ld_boundaries1))])
    state['assignment2'] = np.concatenate([tmp[j][1].astype(int) for j in range(len(ld_boundaries1))])
    
    state['suffstats'] = update_suffstats(state) 
    if not force_shared:
        sample_pi_pop(state)
    sample_V(state) 
    update_p(state) 
    sample_alpha(state) 

    for j in range(len(ld_boundaries1)):
        sample_beta(j, state, idx1_shared=idx1_shared, idx2_shared=idx2_shared, ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, ref_ld_mat1=ref_ld_mat1, ref_ld_mat2=ref_ld_mat2, rho=rho, VS=True)
        compute_varg(j, state, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2) 
        
    state['h2_1'] = np.sum(state['varg1'])
    state['h2_2'] = np.sum(state['varg2'])
    
    state['eta'] = sample_eta(state, ld_boundaries1, ld_boundaries2)




