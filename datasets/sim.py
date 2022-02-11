import numpy as np
from scipy.special import expit
def simulate_data(config, a_int_1, a_int_2):
    h = config["lag"]
    p = config["p"]
    burn = config["burn"]
    T = config["T"]
    n = config["n"]
    noise_A = config["noise_A"]
    noise_Y = config["noise_Y"]
    noise_X = config["noise_X"]
    #function to generate gaussian noise
    def noise(s, sd):
        return np.random.normal(loc=0, scale=sd, size=s)

    #Coefficients
    signs_xa = 2 * np.random.binomial(1, 0.5, h) - np.ones(h)
    coef_ya = np.empty(h)
    coef_xx_rel = np.empty((h-1,p))
    coef_xa = np.empty((h-1,p))
    for i in range(h-1):
        coef_xx_rel[i,:] = np.random.normal(loc=1/(i+2),scale=0.02,size=p)
        coef_xa[i,:] = np.random.normal(loc=1 / (i + 2), scale=0.02, size=p) * signs_xa[i]

    for i in range(h):
        coef_ya[i] = (1/(i+1)) * (-1)**i
    coef_ya = np.flip(coef_ya)
    coef_xx_rel = np.flip(coef_xx_rel)
    coef_xa = np.flip(coef_xa)

    #Errors
    err_X_rel = noise((n, T+h + burn,p),noise_X)
    err_A = noise((n,T+h + burn), noise_A)
    err_Y = noise((n, T + h + burn), noise_Y)

    #Data generating process
    def generate_data(X_rel, A, Y, cf = False):
        start = h
        if cf:
            start = h + burn

        for t in range(start, T + h + burn):
            #Covariates
            X_rel[:, t, :] = err_X_rel[:, t, :]
            for pat in range(n):
                for lag in range(1,h):
                    X_rel[pat, t, :] += coef_xx_rel[lag-1,:] * X_rel[pat, t - lag, :] +0.25* coef_xa[lag-1,:] * (2*A[pat,t-lag]-1)
            X_rel[:, t, :] = np.tanh(X_rel[:, t, :])
            #Treatment
            if cf == False:
                x_contrib = np.mean(X_rel[:,t,:],1 )
                for lag in range(1, h):
                    x_contrib *= np.mean(X_rel[:,t-lag,:],1 )
                x_contrib = np.tan(x_contrib)
                prob_A = expit(x_contrib + err_A[:, t] + (1/p)*Y[:, t-1])
                A[:, t] = np.where(prob_A > 0.5, 1, 0)
            #Outcome
            contrib_y = np.mean(X_rel[:,t,:],1 ) + err_Y[:,t]
            for pat in range(n):
                contrib_y[pat] += 0.5* np.dot(coef_ya, 2*A[pat, t - h+1:t+1] - 1)
            Y[:, t] = contrib_y

    #Factual data
    X_rel_f = noise((n, T+h+burn, p), noise_X)
    #A_f = np.random.binomial(1, 0.5, (n, T + h + burn))
    A_f = np.zeros(((n, T + h + burn)))
    Y_f = noise((n, T + h + burn), noise_Y)
    generate_data(X_rel_f, A_f, Y_f, cf=False)

    #Counterfactual data
    X_rel_cf1 = X_rel_f.copy()
    A_cf1 = A_f.copy()
    A_cf1[:, h + burn:] = np.tile(a_int_1, (n, 1))
    Y_cf1 = Y_f.copy()
    generate_data(X_rel_cf1, A_cf1, Y_cf1, cf=True)

    X_rel_cf2 = X_rel_f.copy()
    A_cf2 = A_f.copy()
    A_cf2[:, h + burn:] = np.tile(a_int_2, (n, 1))
    Y_cf2 = Y_f.copy()
    generate_data(X_rel_cf2, A_cf2, Y_cf2, cf=True)

    #Format data
    d_fact = format_data(X_rel_f, A_f, Y_f, h, burn)
    d_cf1 = format_data(X_rel_cf1, A_cf1, Y_cf1, h, burn)
    d_cf2 = format_data(X_rel_cf2, A_cf2, Y_cf2, h, burn)
    return d_fact, d_cf1, d_cf2


def format_data(X_rel, A, Y, h, burn):
    n = X_rel.shape[0]
    T = X_rel.shape[1] - h - burn
    X_rel = X_rel[:, h+burn:, :]
    A = np.expand_dims(A[:, h + burn:], 2)
    Y = np.expand_dims(Y[:, h+burn:], 2)
    Y_prev = Y.copy()
    Y_prev[:, 0, 0] = np.zeros(n)
    Y_prev[:, 1:, 0] = Y[:, 0:(T - 1), 0]
    data = np.concatenate((Y, A, X_rel, Y_prev), axis=2)
    return data

def generate_interventions(T, treat_nr, dataset):

    if dataset=="sim":
        if treat_nr == 1:
            a_int_1 = np.zeros(T)
            a_int_2 = np.zeros(T)
            a_int_1[5:10] = np.ones(5)
        if treat_nr == 2:
            a_int_1 = np.zeros(T)
            a_int_2 = np.zeros(T)
            a_int_1[3:13] = np.ones(10)
        if treat_nr == 3:
            a_int_1 = np.ones(T)
            a_int_2 = np.zeros(T)
    elif dataset == "mimic":
        if treat_nr == 1:
            a_int_1 = np.zeros(T)
            a_int_1[3:7] = np.ones(4)
            a_int_1[12:15] = np.ones(3)
            a_int_2 = np.zeros(T)
        if treat_nr == 2:
            a_int_1 = np.zeros(T)
            a_int_1[3:7] = np.ones(4)
            a_int_1[10:15] = np.ones(5)
            a_int_2 = np.zeros(T)
        if treat_nr == 3:
            a_int_1 = np.zeros(T)
            a_int_1[1:5] = np.ones(4)
            a_int_1[12:15] = np.ones(3)
            a_int_2 = np.zeros(T)
    else:
        a_int_1 = np.ones(T)
        a_int_2 = np.zeros(T)



    print(f"Treatment 1 is {a_int_1}")
    print(f"Treatment 2 is {a_int_2}")
    return a_int_1, a_int_2