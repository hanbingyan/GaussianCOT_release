import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from scipy.optimize import NonlinearConstraint

# Flatten a symmetric matrix to a vector, diagonal entries first
def inver_sym(A, dim):
    if dim == 1:
        return [A[0, 0]]

    x = []
    for i in range(dim):
        x.append(A[i, i])
    for i in range(dim):
        for j in range(i+1, dim):
            x.append(A[i, j])
    return x

# Construct a symmetric matrix from a vector, diagonal entries first
def sym_matrix(x, dim):
    res = np.zeros((dim, dim))
    for l in range(dim):
        res[l, l] = x[l]
    for k in range(int(dim*(dim-1)/2)):
        i = dim - 2 - int(np.sqrt(-8*k + 4*dim*(dim-1)-7)/2.0 - 0.5)
        j = k + i + 1 - dim*(dim-1)/2 + (dim-i)*((dim-i)-1)/2
        i = int(i)
        j = int(j)
        res[i, j] = x[k+dim]
        res[j, i] = x[k+dim]
    return res

# k-th diagonal entry of D
def kth_diag(psd, k):
    if k == 0:
        return psd[0, 0]
    elif k == 1:
        return psd[1, 1] - psd[0, 1]/psd[0, 0]*psd[1, 0]
    else:
        return psd[k, k] - np.matmul(np.matmul(psd[:k,k], np.linalg.inv(psd[:k, :k])), psd[k, :k])




def optimize(m_dim, n_dim, radius, A, Bp, C, Dp, Sigp, current_obs, pred_mean, maxit, causal=True, robust=True):
    # Bp, Dp are reference model parameters
    if robust==False:
        M = np.matmul(np.matmul(np.matmul(C, A), Sigp), A.T) + np.matmul(C, Bp)
        K = np.matmul(np.matmul(np.matmul(np.matmul(C, A), Sigp), A.T), C.T) + np.matmul(np.matmul(C, Bp), C.T) + Dp

        worst_slope = np.matmul(M.T, np.linalg.inv(K))
        worst_incpt = A @ pred_mean - worst_slope @ C @ A @ pred_mean
        update_mean = worst_slope @ current_obs + worst_incpt
        update_cov = A @ Sigp @ A.T + Bp - M.T @ np.linalg.inv(K) @ M

        return update_mean, update_cov


    def obj(x):
        B = sym_matrix(x[:int((n_dim ** 2 + n_dim) / 2)], n_dim)
        D = sym_matrix(x[int((n_dim ** 2 + n_dim) / 2):int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2)], m_dim)
        Sig = sym_matrix(x[int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2):], n_dim)
        M = np.matmul(np.matmul(np.matmul(C, A), Sig), A.T) + np.matmul(C, B)
        K = np.matmul(np.matmul(np.matmul(np.matmul(C, A), Sig), A.T), C.T) + np.matmul(np.matmul(C, B), C.T) + D
        val = np.trace(np.matmul(np.matmul(A, Sig), A.T) + B) - np.trace(np.matmul(np.matmul(M.T, np.linalg.inv(K)), M))
        return -val

    def cot_cons(x):
        B = sym_matrix(x[:int((n_dim ** 2 + n_dim) / 2)], n_dim)
        D = sym_matrix(x[int((n_dim ** 2 + n_dim) / 2):int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2)], m_dim)
        Sig = sym_matrix(x[int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2):], n_dim)

        # Construct bi-causal OT distance (Corollary 2.2)
        fp1_row = np.concatenate((Bp, np.matmul(Bp, C.T)), axis=1)
        sp2_row = np.concatenate((np.matmul(C, Bp), np.matmul(np.matmul(C, Bp), C.T) + Dp), axis=1)
        Block_BCp = np.concatenate((fp1_row, sp2_row), axis=0)

        f1_row = np.concatenate((B, np.matmul(B, C.T)), axis=1)
        s2_row = np.concatenate((np.matmul(C, B), np.matmul(np.matmul(C, B), C.T) + D), axis=1)
        Block_BC = np.concatenate((f1_row, s2_row), axis=0)

        H = np.eye(n_dim) + np.matmul(A.T, A) + np.matmul(np.matmul(np.matmul(A.T, C.T), C), A)
        COT1 = Block_BCp + Block_BC - 2 * sqrtm(np.matmul(np.matmul(sqrtm(Block_BCp), Block_BC),
                                                          sqrtm(Block_BCp)))
        COT2 = 2 * sqrtm(np.matmul(np.matmul(np.matmul(np.matmul(sqrtm(Sigp), H), Sig), H), sqrtm(Sigp)))
        COT2 = np.matmul(H, Sig) + np.matmul(H, Sigp) - COT2

        res = np.zeros(2 * n_dim + m_dim + 1)
        res[0] = radius - np.trace(COT1).real - np.trace(COT2).real

        # B psd constraint
        for k in range(n_dim):
            res[k + 1] = kth_diag(B, k)

        # D psd constraint
        for k in range(m_dim):
            res[k + n_dim + 1] = kth_diag(D, k)

        # Sig psd constraint
        for k in range(n_dim):
            res[k + n_dim + m_dim + 1] = kth_diag(Sig, k)

        return res


    def ot_cons(x):
        B = sym_matrix(x[:int((n_dim ** 2 + n_dim) / 2)], n_dim)
        D = sym_matrix(x[int((n_dim ** 2 + n_dim) / 2):int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2)], m_dim)
        Sig = sym_matrix(x[int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2):], n_dim)

        # ref variance
        fp1_row = np.concatenate((Sigp, np.matmul(Sigp, A.T), np.matmul(np.matmul(Sigp, A.T), C.T)), axis=1)
        fp2_row = np.concatenate((np.matmul(A, Sigp), np.matmul(np.matmul(A, Sigp), A.T) + Bp,
                                  np.matmul(np.matmul(np.matmul(A, Sigp), A.T), C.T) + np.matmul(Bp, C.T)), axis=1)
        fp3_row = np.concatenate((np.matmul(C, np.matmul(A, Sigp)),
                                  np.matmul(np.matmul(np.matmul(C, A), Sigp), A.T) + np.matmul(C, Bp),
                                  np.matmul(np.matmul(C, Bp), C.T) + D + \
                                  np.matmul(np.matmul(np.matmul(np.matmul(C, A), Sigp), A.T), C.T)), axis=1)
        Ref_var = np.concatenate((fp1_row, fp2_row, fp3_row), axis=0)

        # alter var
        f1_row = np.concatenate((Sig, np.matmul(Sig, A.T), np.matmul(np.matmul(Sig, A.T), C.T)), axis=1)
        f2_row = np.concatenate((np.matmul(A, Sig), np.matmul(np.matmul(A, Sig), A.T) + B,
                                 np.matmul(np.matmul(np.matmul(A, Sig), A.T), C.T) + np.matmul(B, C.T)), axis=1)
        f3_row = np.concatenate((np.matmul(C, np.matmul(A, Sig)),
                                 np.matmul(np.matmul(np.matmul(C, A), Sig), A.T) + np.matmul(C, B),
                                 np.matmul(np.matmul(C, B), C.T) + D + \
                                 np.matmul(np.matmul(np.matmul(np.matmul(C, A), Sig), A.T), C.T)), axis=1)
        Alter_var = np.concatenate((f1_row, f2_row, f3_row), axis=0)

        # trace
        Ref_sq = sqrtm(Ref_var)
        # last term in trace
        three_mul = sqrtm(np.matmul(np.matmul(Ref_sq, Alter_var), Ref_sq))

        # if np.iscomplexobj(three_mul):
            # print('Complex sqrt matrix, max imaginary is', np.max(np.abs(three_mul.imag)))
        TR = np.trace(Ref_var + Alter_var - 2 * three_mul).real

        res = np.zeros(2 * n_dim + m_dim + 1)
        res[0] = radius - TR

        # B psd constraint
        for k in range(n_dim):
            res[k + 1] = kth_diag(B, k)

        # D psd constraint
        for k in range(m_dim):
            res[k + n_dim + 1] = kth_diag(D, k)

        # Sig psd constraint
        for k in range(n_dim):
            res[k + n_dim + m_dim + 1] = kth_diag(Sig, k)

        return res


    ########### Main ###############

    list_Bp = inver_sym(Bp, n_dim)
    list_Dp = inver_sym(Dp, m_dim)
    list_Sigp = inver_sym(Sigp, n_dim)
    init_x = np.array(list_Bp + list_Dp + list_Sigp)

    bnds = ((0, None),)*n_dim + ((None, None),)*int((n_dim**2-n_dim)/2) + ((0, None),)*m_dim + \
           ((None, None),)*int((m_dim**2-m_dim)/2) + ((0, None),)*n_dim + ((None, None),)*int((n_dim**2-n_dim)/2)

    # if causal:
    #     opt_cons = ({'type': 'ineq', 'fun': cot_cons})
    # else:
    #     opt_cons = ({'type': 'ineq', 'fun': ot_cons})

    if causal:
        # opt_cons = ({'type': 'ineq', 'fun': cot_cons})
        opt_cons = NonlinearConstraint(cot_cons, 0, np.inf, keep_feasible=True)
    else:
        opt_cons = NonlinearConstraint(ot_cons, 0, np.inf, keep_feasible=True)

    res = minimize(obj, init_x, method='trust-constr',
                   bounds=bnds, constraints=opt_cons,
                   options={'verbose': 0, 'maxiter': maxit})

    optx = res.x

    B = sym_matrix(optx[:int((n_dim ** 2 + n_dim) / 2)], n_dim)
    D = sym_matrix(optx[int((n_dim ** 2 + n_dim) / 2):int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2)], m_dim)
    Sig = sym_matrix(optx[int((n_dim ** 2 + n_dim) / 2 + (m_dim ** 2 + m_dim) / 2):], n_dim)
    M = np.matmul(np.matmul(np.matmul(C, A), Sig), A.T) + np.matmul(C, B)
    K = np.matmul(np.matmul(np.matmul(np.matmul(C, A), Sig), A.T), C.T) + np.matmul(np.matmul(C, B), C.T) + D

    worst_slope = np.matmul(M.T, np.linalg.inv(K))
    worst_incpt = A @ pred_mean - worst_slope @ C @ A @ pred_mean
    update_mean = worst_slope @ current_obs + worst_incpt
    update_cov =  A @ Sig @ A.T + B - M.T @ np.linalg.inv(K) @ M

    return update_mean, update_cov
