import numpy as np
from .sobol import i4_sobol_generate

def generate_initial(Npar,N_ini,method='sobol'):
    
    ''' Returns N_workers x N_ini x Npar array of initial points '''
    shape = (N_ini,Npar)
    
    if method == 'sobol':
        points = sobol_generate( Npar, N_ini).T
        points = np.reshape(points,shape)
    
    elif method == 'halton':
        points = halton_generate(Npar, N_ini)
        points = np.reshape(points,shape)
        
    elif method == 'runiform':
        points = np.random.uniform(size=shape)

    else:
        print(f'method {method} not available for initial draws construction')
    
    return points

##############
# Sobol draws
##############
def sobol_generate(m,n):

    skip = 1
    return i4_sobol_generate(m,n,skip)


##############
# Halton draws
##############
def halton_generate(dim, n_sample):
    """Halton sequence.

    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return np.array(sample)

def primes_from_2_to(n):
    """Prime number from 2 to n.

    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.

    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence
    