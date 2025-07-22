# MOSES module for python, with simple multiplicative inverse.
# CCK 2018-Aug-15 corrected several bugs.
# CCK 2018-Aug-23 made documentation more pythonic.
# CCK 2018-Nov-19 corrected an error that carried over from my derivation of the normalization. 
#                 The derivation in MultiplicativeInverse.ipynb is now correct.
import numpy as np

def fomod(guess, m, j0):
    """
    MOSES forward model projection, similar to Hans Courrier's fomod.pro.
    guess = 2D array of intensities; first dimension is positions, second is intensities.
    m = spectral order
    j0 = wavelength index associated with line center (unshifted during projection).
    """
    Nx      = guess.shape[0]
    Nlambda = guess.shape[1]
    D = np.zeros(Nx) # Output data (projection) array
    for i in range(Nlambda):
        D += np.roll( guess[:,i], m*(i-j0) )
        
    return D


def multinv(D1, D2, m1, m2, Nlambda, j0, compact=True):
    """
    MOSES multiplicative inverse, using any two projections.
    D1, D2 = projections to be multiplied; to within measurement error, their totals should equal.
    m1, m2 = spectral orders of D1 and D2
    Nlambda = number of elements in wavelength space
    """
    Nx = D1.shape[0] # This had better be a 1D array, asame size as D2. I'm not checking!
    inverse = np.ones((Nx, Nlambda)) # Storage for the inverse
    for i in range(Nlambda):
        inverse[:,i] *= np.roll( D1, m1*(j0-i) )
        inverse[:,i] *= np.roll( D2, m2*(j0-i) )
    #print('diagnostics: moses_mart.py normalization')
    #print(inverse.sum())
    N = np.sqrt( D1.sum() * D2.sum() ) # total counts, geometric mean of two data integrals.
    if compact:
        norm = np.abs(m2-m1) / N
    else:
        norm = N/inverse.sum() # This way always works.
    inverse *= norm 
    print('multinv normalization: ', inverse.sum(), ' out of ', N)
    return inverse

def contrasty(G):
    """
    Modify the guess G by increasing its contrast. 
    Built for the mart Filter keyword.
    """
    p = 0.2 #0.2  # ??? Testing configuration ???
    Gp = G**p
    G2 = G*(1 + Gp) # Move to increase image energy. 
    G2 *= np.sum(G)/np.sum(G2) # Renormalize.
    return G2

def contrast_smooth(G):
    """
    Modify the guess G by increasing its contrast and smoothing. 
    Built for the mart Filter keyword.
    """
    import astropy.convolution as ac

    p = 0.2 #0.2  # ??? Testing configuration ???
    Gp = G**p
    G2 = G*(1 + Gp) # Move to increase image energy.
    kernel = np.array([[0,1,0],[1,4,1],[0,1,0]])
    G2 = ac.convolve(G2, kernel, boundary='extend')
    G2 *= np.sum(G)/np.sum(G2) # Renormalize.
    return G2    

def Ltenth(G):
    """
    Return the L_0.1 norm of G.
    """
    return np.nansum(G**0.1)


def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array

    stolen from https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
    '''
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def negentropy(G):
    """
    Return the negative entropy of sqrt(G)
    
    I use the negative entropy because I want a function to maximize, and
    I actually want to minimize the entropy of spectra to concentrate the
    intensity into as few features (spectral lines) as possible. I view this
    as a rough moral equivalent to minimizing the "L0 norm".
    
    The rationale behind the square root is that if G is in counts, then the
    unit bins of the image histogram in sqrt(G) will be spaced at the 1-sigma 
    uncertainty across all signal levels.
    """
    return -entropy(np.ndarray.flatten(np.sqrt(G).round()))
        # flatten: make the array 1D so entropy() can handle it.
        # round: round to nearest unit, so the intensities are grouped in discrete integer bins.

def zpad(I,Npad):
    """
    Zero pad an x,lambda array
    
    Arguments:
    I = Numpy ndarray (Nx, Nlambda)
    Npad = integer; number of zeros to add at both sides of the array
    
    Returned data shape will be (Nx, Nlambda + 2*Npad)
    """
    Nx = I.shape[0]
    Nlambda = I.shape[1]
    result = np.zeros((Nx, Nlambda + 2*Npad))
    result[:, Npad:Npad+Nlambda] = I
    return result

def window(shape):
    """
    Create a rectangular array of specified shape, with maximum value unity and
    parabolic (Welch-like) windowing along the rows. Note that the window is
    crafted so that the zeroes of the parabola are one element beyond the edges
    of the array, so there are no zero elements.
    """
    (Nx,Nlambda) = shape
    x,Lambda = np.meshgrid(np.arange(Nx), np.arange(Nlambda), indexing='ij')
    result = (Lambda+1)*(Nlambda - Lambda)
    return result/np.amax(result)
    

def antialias(D):
    """
    Anti-alias a 1D MOSES data array. This is a simple smoothing filter that MART
    will apply automatically to the outboard orders. Note the periodic boundary,
    which is consistent with MART.
    """
    import astropy.convolution as ac
    return ac.convolve(D, [0.25,0.5,0.25], boundary='wrap')




def mart(DataArr, mArr, Nlambda, j0, 
         maxiter=40, AntiAlias=True, Verbose=True, InitGuess=None, 
         maxouter=1, Filter=None, Maximize=None, LGOF=False):
    """
    MART: NEW VERSION WITH OPTIONAL LOCALIZED GOF (LGOF)
    
    Multiplicative Algebraic Reconstruction Technique, using MOSES-style projections into an
    arbitrary number of spectral orders. Infinite order is not (yet?) supported, but an
    initial guess based on the infinite order can be used.
    
    When used without the maxouter, Filter, or Maximize keywords, this is the simplest
    possible MART. Up to maxiter multiplicative corrections are performed. The loop
    exits when the reduced "chi square" drops below unity. I say "chi square" in quotes
    because DoFs are not counted honestly (how do you do that when the inversion has
    more degress of freedom than the data?). If we reach maxiter iterations without
    convergence of chisquare, an error is thrown.
    
    The maxouter and Filter keywords are designed to allow iterative filtering in an
    outer loop ("outeration") that encloses the MART iterations described above. Before
    each run of MART, the guess is put through the specified Filter function. 
    The maxouter value gives the maximum number of outerations.
    
    The Maximize keyword adds a convergence criterion to the outer loop. Supply the
    name of some figure of merit, perhaps a "norm," of the guess. The concept
    here is that the Filter drives the "norm" to increase until its maximum is
    reached. If the value of "norm" is seen to decrease, this is interpreted as
    having reached the maximum, and we terminate the outerations.
        
    Arguments: 
    DataArr = (Nx,Nm) array of data in Nm projections.
    mArr = (Nm) array of integer spectral orders.
    Nlambda = integer number of wavelength pixels in the inversion.
    j0 = index of the nominal line center wavelength. This column of the spectrum
        will not be shifted when projecting in the forward model (see fomod).
    
    Keywords:
    maxiter = maximum number of (inner) multiplicative inverse iterations allowed to 
        get the reduced "chi square" down below unity.
    AntiAlias = if true, then antialias filtering is done on the nonzero spectral
        orders. Recommended for numerical stability, and to prevent the
        nonintersection anomaly which can sometimes cause horrible aliasing and
        unusually bad plaid artifacts.
    InitGuess = initial guess (by default, a flat initial guess is used). This is
        useful either for handing off a guess from a previous inversion effort
        or for other preconditioning of the inversion result.
    maxouter = maximum number of outer loop iterations ("outerations").
    Filter = filter function for iterative filtering of the guess.
    Maximize = criterion ("norm") function for evaluating the guess and judging
        convergence of the outerations.
    """
    
    Nm = mArr.size
    if (len(DataArr.shape) != 2):
        print('DataArr.shape = ',DataArr.shape,' len(DataArr.shape) = ',len(DataArr.shape))
        raise Exception('DataArr must be 2D (Nx,Nm).')
    if (DataArr.shape[0] != Nm):
        raise Exception('First dimension of DataArr must be same size as mArr.')
    Nx = DataArr.shape[1]
    
    # Anti-aliasing (recommended, fixes the nonintersection anomaly).
    DataArr2 = DataArr.copy()
    if AntiAlias:
        for m in range(Nm):
            if (mArr[m] != 0): # Apply anti-alias filter only to the nonzero spectral orders.
                DataArr2[m,:] = antialias(DataArr2[m,:])
    
    # Initial Guess (default uniform; or forwarded from a previous run, or an opportunity for pre-conditioning)
    if (InitGuess is None): # note use of 'is' rather than '==' (identity rather than equality)
        guess = np.ones((Nx,Nlambda)) # initialize guess
        guess *= DataArr2.sum()/Nm/guess.sum() # Normalize guess using mean of data array.
    else:
        guess = InitGuess.copy()
    
    norm_history = np.empty(maxouter) # Storage for the history of "norm" values.
    norm_history.fill(np.nan) # Initialize with nan, to avoid confusion with calculated "norm" values
    for iouter in range(maxouter): # Outer inversion loop, with regularization options.
        if (Maximize != None): # Before outerating, let's evaluate the data norm and see if we are done.
            norm_history[iouter] = Maximize(guess) # Calculate the "norm" for this outeration.
            print('Maximize(Guess) = ', norm_history[iouter])
            if (norm_history[iouter] == np.nanmax(norm_history)): # Is this the best we've seen?
                best_guess = guess   # Save this guess
                best_iouter = iouter # Remember which outeration we're on
        print('outeration ',iouter+1,' of, at most, ', maxouter)
        if (Filter != None):
            guess = Filter(guess)

        for i in range(maxiter): # Inner inversion loop, pure MART, adjusts guess to globally fit the data.
            Nconverged = Nm # Assume all (!) of the Nm orders have converged to chisq <= 1.
            if Verbose:
                print('MART iteration ',i+1,' of, at most, ', maxiter)
            for m in range(Nm):
                Dprime = fomod(guess, mArr[m], j0)
                chisq = np.mean( (Dprime - DataArr2[m,:])**2 / (1 + Dprime) )
                    # Notes on "reduced chisquare"
                    #    (0) assumes data units are 'counts', with shot noise.
                    #    (1) assumes read noise of order 1 count: sigma**2 = 1+Dprime.
                    #    (2) normalizes by number of data elements, not proper DoF. 
                if Verbose:
                    print(mArr[m],' order "reduced chisquare" = ',chisq)
                if (chisq > 1.):
                    Nconverged -= 1
                    correction = (DataArr2[m,:]/Dprime)**(2./Nm)
                    for j in range(Nlambda):
                        guess[:,j] *= np.roll( correction, mArr[m]*(j0-j) )
            if (Nconverged==3): # if true, then all 3 chisq values are < 1. We deem the inversion converged.
                break #this concludes the iteration; go to the next outeration!
        if (Nconverged != 3):
            raise ValueError('MART failed to converge; maximum number of iterations exceeded!')

        if LGOF:  # Optional second inner MART loop, using localized goodness-of-fit (LGOF).
            for i in range(maxiter):
                Nconverged = Nm # Assume all (!) of the Nm orders have converged to chisq <= 1.
                if Verbose:
                    print('MART LGOF iteration ',i+1,' of, at most, ', maxiter)
                for m in range(Nm):
                    Dprime = fomod(guess, mArr[m], j0)
                    GOF = (Dprime - DataArr2[m,:])**2 / (1 + Dprime) # pixel-wise goodness of fit
                    chisq = np.mean(GOF)
                        # Notes on "reduced chisquare"
                        #    (0) assumes data units are 'counts', with shot noise.
                        #    (1) assumes read noise of order 1 count: sigma**2 = 1+Dprime.
                        #    (2) normalizes by number of data elements, not proper DoF. 
                    if Verbose:
                        print(mArr[m],' order "reduced chisquare" = ',chisq)
                    if (any(GOF > 1.)): # If even 1 pixel has error greater than 1 sigma...
                        Nconverged -= 1
                        correction = (DataArr2[m,:]/Dprime)**((GOF>1.)*2./Nm)
                            # Note that corrections are unity where GOF <= 1.
                        for j in range(Nlambda):
                            guess[:,j] *= np.roll( correction, mArr[m]*(j0-j) )
                if (Nconverged==3): # if true, then all 3 channels are converged.
                    break #this concludes the iteration; go to the next outeration!
            if (Nconverged != 3):
                raise ValueError('LGOF MART failed to converge; max # iterations exceeded!')
    
    if (Maximize != None):
        print('Best result, Maximize(Guess) = ',norm_history[best_iouter],', was obtained on outeration ',best_iouter)
        return best_guess
    else:
        return guess
    
    