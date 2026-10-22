"""
Script for The Art of Scientific Computing:
A Matter of Gaussian Approximation.

Produces the figures, tables and verification checks seen in the report.
See the README.md for installation, execution instructions, and further commentary.

Author: Ava F. Piper
Date: 22 October 2026
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import quad
import scipy


# Main settings (atomic units)

n_basis = 12
alpha_start = 0.01                  # bohr^-2
beta_ratio = 1.9
max_charge = 25
condition_limit = 1e12       # chosen numerical screening criterion
figure_dir = Path(__file__).resolve().parent / 'Figures'
report_dir = Path(__file__).resolve().parent / 'Report_Tables'
show_figures = True          # Display figures as well as saving them
figure_names = [
    'Chapter1_Panel.png',
    '01_hydrogen_wavefunction.png',
    '02_hydrogen_radial_probability.png',
    '03_convergence_and_conditioning.png',
    '04_alpha_beta_heatmap.png',
    '05_fixed_vs_scaled_charge.png',
    '06_manganese_radial_comparison.png',
    '07_scaling_collapse.png',
    '08_nuclear_cusp.png',
    '09_local_energy.png',
    '10_energy_vs_wavefunction_error.png',
    '11_basis_pruning.png',
]
figure_number = 0
saved_figures = []

# Search a fixed grid of exponents for each basis size.
# These are the best tested choices, not an exact optimum.
search_sizes = range(2, 21)
search_alphas = np.logspace(-3, -1, 13)
search_betas = np.linspace(1.4, 2.8, 15)
target_errors = (1e-2, 1e-3, 1e-4)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 110,
})


# Gaussian basis and matrix elements
# form all pairs of exponents at once with NumPy broadcasting, rather than
# looping over integrals. SciPy's eigh only solves for the lowest state.

# The separate tests below check that these shortcuts are correct.

def matrices(n=n_basis, alpha=alpha_start, beta=beta_ratio, z=1):
    """Return (exponents, norms, overlap S, Hamiltonian H) for n Gaussians."""
    if n < 1 or alpha <= 0 or beta <= 1 or z <= 0:
        raise ValueError('Require n>=1, alpha>0, beta>1 and z>0')

    a = alpha * beta**np.arange(n)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError('Gaussian exponents must be positive and finite')
    norms = (2*a/np.pi)**0.75
    p = a[:, None] + a[None, :]
    factors = norms[:, None] * norms[None, :]

    # From the Gaussian basis treatment in the Quiney and Flynn course notes.
    # Using the analytic forms is much faster for the grid search.
    # S = overlap, T = kinetic energy, V = electron-nucleus attraction.
    S = factors * (np.pi/p)**1.5
    T = 3*a[:, None]*a[None, :]/p * S
    V = -2*np.pi*z*factors/p
    return a, norms, S, T + V


def solve(n=n_basis, alpha=alpha_start, beta=beta_ratio, z=1):
    """Return (energy in Ha, coefficients, exponents, norms, cond(S))."""
    a, norms, S, H = matrices(n, alpha, beta, z)
    # We only need the ground state, so there is no reason to find every eigenpair.
    values, vectors = eigh(H, S, subset_by_index=[0, 0])
    c = vectors[:, 0]
    if np.dot(c, norms) < 0:       # choose a consistent wavefunction sign
        c = -c
    return values[0], c, a, norms, np.linalg.cond(S)


def wavefunction(r, c, a, norms):
    """Add the Gaussian functions together at the requested radii."""
    r = np.asarray(r)
    return np.sum((c*norms)[:, None] *
                  np.exp(-a[:, None]*r[None, :]**2), axis=0)


def radial_probability(r, c, a, norms):
    """Calculate the radial probability from the wavefunction."""
    return 4*np.pi*r**2*wavefunction(r, c, a, norms)**2


def exact_wavefunction(r, z=1):
    """Exact hydrogenic 1s wavefunction for comparison."""
    return np.sqrt(z**3/np.pi)*np.exp(-z*np.asarray(r))


def exact_radial(r, z=1):
    """Exact radial probability for comparison."""
    r = np.asarray(r)
    return 4*z**3*r**2*np.exp(-2*z*r)


def relative_error(energy, z=1):
    """Compare the calculated energy with the exact value -Z**2/2."""
    exact_energy = -z**2/2
    return abs((energy-exact_energy)/exact_energy)


def mean_radius(c, a, norms):
    """Calculate the mean radius using the Gaussian integral from the report."""
    p = a[:, None] + a[None, :]
    R = 2*np.pi*norms[:, None]*norms[None, :]/p**2
    return c @ R @ c


def show():
    """Save each plot in the Figures folder and display it if requested."""
    global figure_number
    if figure_number >= len(figure_names):
        raise RuntimeError('More figures produced than expected')
    fig = plt.gcf()
    # The first figure already uses constrained layout for its colourbar.
    if figure_number != 0 and len(fig.axes) == 1:
        fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / figure_names[figure_number]
    fig.savefig(filename, dpi=250, bbox_inches='tight')
    figure_number += 1
    saved_figures.append(filename.name)
    if show_figures:
        plt.show()
    plt.close(fig)


def heading(title):
    """Print a heading to separate the terminal results."""
    print('\n' + '='*72)
    print(title)
    print('='*72)



def relative_eigen_residual(H, S, energy, c):
    """Return the scaled residual of Hc = ESc (smaller is better)."""
    numerator = np.linalg.norm(H @ c - energy*S @ c)
    denominator = np.linalg.norm(c)*(np.linalg.norm(H, 2)
                                      + abs(energy)*np.linalg.norm(S, 2))
    return numerator/denominator


def verify_matrix_integrals():
    """Compare some matrix elements with numerical integration."""
    heading('INDEPENDENT MATRIX-ELEMENT QUADRATURE')
    a, norms, S, H = matrices()
    worst = 0.0
    for i, j in ((0, 0), (0, 4), (3, 7), (11, 11)):
        ai, aj = a[i], a[j]
        factor = norms[i]*norms[j]
        p = ai + aj
        overlap = 4*np.pi*factor*quad(lambda r: r*r*np.exp(-p*r*r),
                                      0, np.inf, epsabs=1e-12)[0]
        kinetic = 8*np.pi*ai*aj*factor*quad(
            lambda r: r**4*np.exp(-p*r*r), 0, np.inf, epsabs=1e-12)[0]
        potential = -4*np.pi*factor*quad(
            lambda r: r*np.exp(-p*r*r), 0, np.inf, epsabs=1e-12)[0]
        t_exact = 3*ai*aj/p*S[i, j]
        v_exact = H[i, j]-t_exact
        for name, numeric, analytic in (('S', overlap, S[i, j]),
                                         ('T', kinetic, t_exact),
                                         ('V', potential, v_exact)):
            difference = abs(numeric-analytic)
            worst = max(worst, difference/max(abs(analytic), 1e-14))
            assert np.isclose(numeric, analytic, rtol=1e-9, atol=1e-12), (
                f'Integral mismatch: {name}[{i},{j}]')
    print(f'S, T and V: 12 independent comparisons PASS; maximum relative difference = {worst:.3e}')
    return worst


# Check the calculations before making figures

def validate():
    """Check the matrix calculations before making the plots."""
    heading('NUMERICAL VALIDATION')
    a, norms, S, H = matrices()
    e, c, _, _, k = solve()
    assert np.allclose(S, S.T, rtol=1e-12, atol=1e-12)
    assert np.allclose(H, H.T, rtol=1e-12, atol=1e-12)
    assert np.isclose(c @ S @ c, 1, atol=1e-11)
    assert np.isclose(c @ H @ c, e, atol=1e-10)
    residual = relative_eigen_residual(H, S, e, c)
    assert residual < 1e-10
    assert np.allclose(np.diag(S), 1.0, atol=1e-12)
    assert e >= -0.5-1e-10
    print('Matrix symmetry, normalisation, eigenpair residual and variational bound: PASS')
    print(f'Relative eigenpair residual = {residual:.3e} (limit 1e-10)')
    print(f'Condition number times machine epsilon = {k*np.finfo(float).eps:.3e}')
    verify_matrix_integrals()

    # Check the expected Z-scaling directly using the matrices:
    # scaling the exponents by Z^2 should leave S unchanged,
    # while H should increase by Z^2.
    _, _, S25, H25 = matrices(alpha=alpha_start*25**2, z=25)
    assert np.allclose(S25, S, rtol=1e-10, atol=1e-10)
    assert np.allclose(H25/25**2, H, rtol=1e-10, atol=1e-10)
    for z in (1, 2, 5, 10, 25, 100):
        ez, _, _, _, kz = solve(alpha=alpha_start*z*z, z=z)
        assert np.isclose(ez/z**2, e, rtol=1e-9, atol=1e-10)
        assert np.isclose(kz, k, rtol=1e-8)
        assert ez >= -z**2/2-1e-7
        print(f'Z={z:3d}  E/Z^2={ez/z**2: .10f} Ha  '
              f'|E/Z^2-E(1)|={abs(ez/z**2-e):.2e} Ha')
    print('Hydrogenic scaling and condition-number invariance: PASS')
    print('Scaling tolerance: energy rtol=1e-9, atol=1e-10 Ha; conditioning rtol=1e-8')


# Introductory hydrogenic overview

def plot_overview():
    """Make the hydrogenic overview figure for Chapter 1."""
    r = np.linspace(0, 16, 1200)
    R1 = 2*np.exp(-r)
    R2s = (2-r)*np.exp(-r/2)/(2*np.sqrt(2))
    R2p = r*np.exp(-r/2)/(2*np.sqrt(6))
    P1, P2s, P2p = r*r*R1*R1, r*r*R2s*R2s, r*r*R2p*R2p
    P1 /= P1.max()
    P2s /= P2s.max()
    P2p /= P2p.max()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    red_box = dict(boxstyle='round,pad=0.2', facecolor='white',
                   edgecolor='red', lw=1.4)
    arrow = dict(arrowstyle='->', color='0.45', lw=1.1,
                 shrinkA=4, shrinkB=3)

    ax[0].set(xlim=(0, 1), ylim=(0, 5.5))
    ax[0].axis('off')
    ax[0].annotate('', xy=(.15, 5.15), xytext=(.15, .3),
                   arrowprops=dict(arrowstyle='-|>', color='0.35',
                                   lw=2.5, mutation_scale=22))
    ax[0].text(.05, 2.75, 'Energy', rotation=90, ha='center', va='center')
    for n, y in {1: .85, 2: 2.65, 3: 3.8, 4: 4.4, 5: 4.8}.items():
        ax[0].hlines(y, .35, .65, color='purple' if n == 1 else '0.6', lw=3)
        ax[0].text(.73, y, r'$1s$' if n == 1 else rf'$n={n}$',
                   va='center', bbox=red_box if n == 1 else None)
    ax[0].set_title('(a) Hydrogenic Energy Levels')

    ax[1].plot(r, P1, color='purple', lw=2.5)
    ax[1].plot(r, P2p, color='tab:red', lw=2.5)
    ax[1].plot(r, P2s, color='tab:green', lw=2.5)
    # Label the three radial curves and highlight the 1s state.
    for name, point, textpoint, box in [
        (r'$1s$', (1, 1), (1, 1.18), red_box),
        (r'$2p$', (4, 1), (3.8, 1.18), None),
        (r'$2s$', (3+np.sqrt(5), 1), (6.4, 1.18), None),
    ]:
        ax[1].annotate(name, xy=point, xytext=textpoint,
                       ha='center', va='bottom', fontsize=13,
                       bbox=box, arrowprops=arrow)
    ax[1].set(xlim=(0, 16), ylim=(0, 1.34),
              xlabel=r'Distance From Nucleus, $r$ (bohr)',
              ylabel='Relative Radial Probability')
    ax[1].set_title('(b) Hydrogenic Radial Distributions')

    x = np.linspace(-4, 4, 350)
    X, Y = np.meshgrid(x, x)
    density = np.exp(-2*np.sqrt(X*X + Y*Y))
    density /= density.max()
    im = ax[2].imshow(density, extent=(-4, 4, -4, 4),
                      origin='lower', cmap='inferno', interpolation='bilinear')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_aspect('equal')
    ax[2].set_title(r'(c) $1s$ Electron Density')
    fig.colorbar(im, ax=ax[2], fraction=.046, pad=.04,
                 label='Relative Electron Density')
    show()


# Hydrogen benchmark: wavefunction and radial probability

def plot_hydrogen(e, c, a, norms):
    """Compare the hydrogen wavefunction and radial probability with the exact result."""
    r = np.linspace(0, 8, 1200)
    calculated = wavefunction(r, c, a, norms)
    exact = exact_wavefunction(r)

    fig, ax = plt.subplots(2, 1, figsize=(7.6, 5.7), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1], 'hspace': .08})
    ax[0].plot(r, exact, 'k--', lw=2, label='Exact')
    ax[0].plot(r, calculated, lw=2, label='Gaussian basis')
    ax[0].set_ylabel(r'$\psi(r)$ (bohr$^{-3/2}$)')
    ax[0].legend()
    ax[1].plot(r, calculated-exact, lw=1.8)
    ax[1].axhline(0, color='k', lw=.7)
    ax[1].set(xlabel=r'$r$ (bohr)', ylabel=r'$\Delta\psi$', xlim=(0, 8))
    show()

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(r, exact_radial(r), 'k--', lw=2, label='Exact')
    ax.plot(r, radial_probability(r, c, a, norms), lw=2, label='Gaussian basis')
    ax.axvline(1.5, color='k', ls=':', label=r'Exact $\langle r\rangle$')
    ax.axvline(mean_radius(c, a, norms), color='tab:blue', ls=':',
               label=r'Gaussian $\langle r\rangle$')
    ax.set(xlabel=r'$r$ (bohr)', ylabel=r'$P(r)$ (bohr$^{-1}$)', xlim=(0, 8))
    ax.legend()
    show()


# Basis size, convergence and conditioning

def plot_convergence():
    """See how energy error and conditioning change as the basis grows."""
    ns = np.arange(2, 31)
    fig, ax = plt.subplots(2, 1, figsize=(7.6, 6.3), sharex=True,
                           gridspec_kw={'hspace': .08})
    for beta in (1.6, 1.9, 2.3):
        errors, conditions = [], []
        for n in ns:
            try:
                energy, _, _, _, condition = solve(n=n, beta=beta)
                errors.append(relative_error(energy))
                conditions.append(condition)
            except (ValueError, np.linalg.LinAlgError) as err:
                print(f'Convergence calculation failed: N={n}, beta={beta}: {err}')
                errors.append(np.nan)
                conditions.append(np.nan)
        line, = ax[0].semilogy(ns, errors, 'o-', ms=3.5,
                              label=rf'$\beta={beta}$')
        ax[1].semilogy(ns, conditions, 'o-', ms=3.5, color=line.get_color())
    ax[0].set_ylabel('Relative energy error')
    ax[0].legend()
    ax[1].axhline(condition_limit, color='k', ls='--',
                  label='Chosen conditioning limit')
    ax[1].set(xlabel='Number of Gaussians $N$', ylabel=r'$\kappa(S)$',
              yscale='log', xlim=(2, 30))
    ax[1].legend(loc='lower right')
    show()


# Exponent parameter map

def plot_parameter_map():
    """Plot the energy error for different alpha and beta choices at N=12."""
    alphas = np.logspace(-3, -1, 61)
    betas = np.linspace(1.35, 3.05, 61)
    errors = np.full((len(betas), len(alphas)), np.nan)
    failed = 0
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            try:
                energy, _, _, _, _ = solve(alpha=alpha, beta=beta)
                errors[i, j] = np.log10(max(relative_error(energy), 1e-16))
            except (ValueError, np.linalg.LinAlgError) as err:
                failed += 1
                print(f'  Parameter-map failure: alpha={alpha:.3g}, beta={beta:.3g}: {err}')
    print(f'Parameter-map eigensolver failures: {failed}')
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    im = ax.pcolormesh(alphas, betas, errors, shading='auto', cmap='viridis')
    ax.set_xscale('log')
    ax.plot(alpha_start, beta_ratio, 'rx', ms=10, mew=2.5, label='Initial choice')
    ax.set(xlabel=r'$\alpha$ (bohr$^{-2}$)', ylabel=r'$\beta$')
    ax.legend()
    fig.colorbar(im, ax=ax, label=r'$\log_{10}$(relative energy error)')
    show()


# Find a basis for each target accuracy
def basis_selection():
    """Find the smallest tested N that reaches each energy-error target."""
    heading('BASIS SELECTION (discrete parameter grid)')
    print(f'Grid: N=2..20, {len(search_alphas)} alpha values, '
          f'{len(search_betas)} beta values, kappa(S) <= {condition_limit:.0e}')
    print('Target       N       alpha       beta      actual error       kappa(S)')
    choices = {target: None for target in target_errors}
    failed = 0
    attempted = 0
    for n in search_sizes:
            # Compare the choices at this N before moving to a larger basis.
        candidates = []
        for alpha in search_alphas:
            for beta in search_betas:
                attempted += 1
                try:
                    energy, _, _, _, condition = solve(n=n, alpha=alpha, beta=beta)
                except (ValueError, np.linalg.LinAlgError) as err:
                    failed += 1
                    print(f'  Basis-search failure: N={n}, alpha={alpha:.3g}, beta={beta:.3g}: {err}')
                    continue
                error = relative_error(energy)
                if condition <= condition_limit:
                    candidates.append((error, n, alpha, beta, condition))
        if not candidates:
            continue
        best = min(candidates)  # smallest energy error at this basis size
        for target in target_errors:
            if choices[target] is None and best[0] <= target:
                choices[target] = best
        # Once every target has been reached, larger bases are not needed here.
        if all(value is not None for value in choices.values()):
            break
    for target, result in choices.items():
        if result is None:
            print(f'{target:8.0e}    Not achieved on tested grid')
        else:
            err, n, alpha, beta, condition = result
            print(f'{target:8.0e}    {n:2d}   {alpha:10.5g}   {beta:7.3f}   '
                  f'{err:12.4e}   {condition:12.4e}')
    print(f'Grid-search solves attempted: {attempted}')
    print(f'Failed grid-search solves: {failed}')
    report_dir.mkdir(exist_ok=True)
    with (report_dir / 'basis_selection.csv').open('w') as file:
        file.write('target_error,N,alpha,beta,actual_error,condition_number\n')
        for target, result in choices.items():
            if result is not None:
                err, n, alpha, beta, condition = result
                file.write(f'{target},{n},{alpha:.12g},{beta:.12g},{err:.12g},{condition:.12g}\n')
    return choices


# Transfer the hydrogen basis to other nuclear charges

def plot_charge_transfer():
    """Compare using the hydrogen basis unchanged and scaling it with Z."""
    zs = np.arange(1, max_charge+1)
    fixed, scaled = [], []
    heading('NUCLEAR-CHARGE TRANSFER')
    print(' Z      fixed error      scaled error       E_scaled (Ha)')
    rows = []
    for z in zs:
        ef, _, _, _, _ = solve(z=int(z))
        es, _, _, _, _ = solve(alpha=alpha_start*z*z, z=int(z))
        fixed.append(relative_error(ef, z))
        scaled.append(relative_error(es, z))
        rows.append((z, fixed[-1], scaled[-1], ef, es))
        if z in (1, 2, 5, 10, 25):
            print(f'{z:2d}    {fixed[-1]:12.6e}   {scaled[-1]:12.6e}   {es:14.8f}')
    report_dir.mkdir(exist_ok=True)
    np.savetxt(report_dir / 'charge_transfer.csv', rows, delimiter=',',
               header='Z,fixed_relative_error,scaled_relative_error,fixed_energy_Ha,scaled_energy_Ha',
               comments='', fmt=['%d', '%.12e', '%.12e', '%.12e', '%.12e'])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.semilogy(zs, fixed, 'o-', ms=3.5, label='Fixed hydrogen exponents')
    ax.semilogy(zs, scaled, 's-', ms=3.5, label=r'$Z^2$-scaled exponents')
    ax.set(xlabel='Nuclear charge $Z$', ylabel='Relative energy error',
           xlim=(.5, max_charge+.5))
    ax.legend()
    show()

    z = 25
    r = np.linspace(0, .55, 1300)
    _, cf, af, nf, _ = solve(z=z)
    _, cs, ass, nss, _ = solve(alpha=alpha_start*z*z, z=z)
    exact = exact_radial(r, z)
    pf = radial_probability(r, cf, af, nf)
    ps = radial_probability(r, cs, ass, nss)
    fig, ax = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1], 'hspace': .08})
    ax[0].plot(r, exact, 'k--', lw=2, label='Exact')
    ax[0].plot(r, pf, lw=2, label='Fixed exponents')
    ax[0].plot(r, ps, lw=2, label=r'$Z^2$-scaled exponents')
    ax[0].set_ylabel(r'$P(r)$ (bohr$^{-1}$)')
    ax[0].legend()
    ax[1].plot(r, pf-exact, lw=1.8, label='Fixed residual')
    ax[1].plot(r, ps-exact, lw=1.8, label='Scaled residual')
    ax[1].axhline(0, color='k', lw=.7)
    ax[1].set(xlabel=r'$r$ (bohr)', ylabel=r'$\Delta P(r)$', xlim=(0, .55))
    ax[1].legend(fontsize=9)
    show()


# Compare scaled radial distributions

def plot_scaling_collapse():
    """Check whether the scaled radial distributions lie on top of each other.

    Output: Figures/07_scaling_collapse.png.
    """
    rho = np.linspace(0, 8, 1200)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for z in (1, 2, 5, 10, 25):
        _, c, a, norms, _ = solve(alpha=alpha_start*z*z, z=z)
        ax.plot(rho, radial_probability(rho/z, c, a, norms)/z,
                lw=1.8, label=rf'$Z={z}$')
    ax.plot(rho, 4*rho**2*np.exp(-2*rho), 'k--', lw=2, label='Exact')
    ax.set(xlabel=r'$\rho=Zr$', ylabel=r'$P_Z(r)/Z$', xlim=(0, 8))
    ax.legend(ncol=2)
    show()


# Near-nuclear behaviour and wavefunction errors

def derivative(r, c, a, norms):
    """Differentiate the Gaussian expansion with respect to r."""
    r = np.asarray(r)
    return np.sum((-2*a*c*norms)[:, None] * r[None, :] *
                  np.exp(-a[:, None]*r[None, :]**2), axis=0)


def laplacian(r, c, a, norms):
    """Calculate the Laplacian needed for the local energy."""
    r = np.asarray(r)
    return np.sum((c*norms)[:, None] *
                  (4*a[:, None]**2*r[None, :]**2 - 6*a[:, None]) *
                  np.exp(-a[:, None]*r[None, :]**2), axis=0)


def plot_nuclear_cusp(c, a, norms):
    """Compare the wavefunctions and their slopes near the nucleus."""
    heading('NUCLEAR CUSP')
    r = np.linspace(0, 1.2, 800)
    exact = exact_wavefunction(r)
    gaussian = wavefunction(r, c, a, norms)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(r, exact, 'k--', lw=2, label='Exact (Slater 1s)')
    ax[0].plot(r, gaussian, lw=2, label='Gaussian basis')
    ax[0].set(xlabel=r'$r$ (bohr)', ylabel=r'$\psi(r)$ (bohr$^{-3/2}$)',
              title='(a) Near-nucleus wavefunction')
    ax[0].legend()
    ax[1].plot(r, -exact, 'k--', lw=2, label='Exact derivative')
    ax[1].plot(r, derivative(r, c, a, norms), lw=2, label='Gaussian derivative')
    ax[1].set(xlabel=r'$r$ (bohr)', ylabel=r'$d\psi/dr$ (bohr$^{-5/2}$)',
              title='(b) Nuclear cusp')
    ax[1].legend()
    print(f'Exact psi(0)     : {exact[0]:.9f}')
    print(f'Gaussian psi(0)  : {gaussian[0]:.9f}')
    print(f"Exact psi'(0)   : {-exact[0]:.9f}")
    print(f"Gaussian psi'(0): {derivative(np.array([0.]), c, a, norms)[0]:.9f}")
    show()


def plot_local_energy(c, a, norms):
    """Plot the local energy to show the behaviour near the nucleus."""
    r = np.linspace(.015, 2.5, 1000)
    g = wavefunction(r, c, a, norms)
    local = -.5*laplacian(r, c, a, norms)/g - 1/r
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, local, lw=2, label='Gaussian local energy')
    ax.axhline(-.5, color='k', ls='--', label='Exact local energy')
    ax.set(xlabel=r'$r$ (bohr)', ylabel='Local energy (Ha)',
           ylim=(-12, 3), title='Local energy of the Gaussian approximation')
    ax.legend()
    show()


def l2_error(c, a, norms):
    """Return the 3D L2 wavefunction error, integrating over radial intervals."""
    def squared_difference(r):
        gaussian = np.dot(c*norms, np.exp(-a*r*r))
        exact = np.exp(-r)/np.sqrt(np.pi)
        return 4*np.pi*r*r*(gaussian-exact)**2

    intervals = [(0, .1), (.1, .5), (.5, 2), (2, 8), (8, 30), (30, 80)]
    pieces = [quad(squared_difference, low, high, epsabs=1e-18,
                   epsrel=1e-9, limit=250)[0] for low, high in intervals]
    return np.sqrt(max(0., sum(pieces)))


def l2_check(c, a, norms):
    """Independently calculate the L2 error using a fine radial grid."""
    r = np.linspace(0, 40, 120001)
    diff = wavefunction(r, c, a, norms) - exact_wavefunction(r)
    return np.sqrt(np.trapezoid(4*np.pi*r*r*diff*diff, r))


def plot_wavefunction_convergence():
    """Compare energy convergence with wavefunction convergence."""
    heading('ENERGY AND WAVEFUNCTION CONVERGENCE')
    ns = np.arange(2, 31)
    energy_errors, wave_errors = [], []
    results = {}
    for n in ns:
        energy, c, a, norms, condition = solve(n=int(n))
        wave_error = l2_error(c, a, norms)
        energy_error = relative_error(energy)
        energy_errors.append(energy_error)
        wave_errors.append(wave_error)
        results[n] = (energy_error, wave_error, condition)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(ns, energy_errors, 'o-', ms=4, label='Relative energy error')
    ax.semilogy(ns, wave_errors, 's-', ms=4, label=r'Wavefunction $L^2$ error')
    ax.set(xlabel='Number of Gaussians $N$', ylabel='Error',
           title='Global energy versus wavefunction accuracy')
    ax.legend()
    print(' N    energy error       L2 error          kappa(S)')
    for n in (4, 8, 12, 20, 30):
        er, wr, k = results[n]
        print(f'{n:2d}    {er:12.3e}    {wr:12.3e}    {k:12.3e}')
    # Check the N=30 L2 error a second way. Agreement checks the integral,
    # but does not explain why the wavefunction error levels off.
    _, c30, a30, norms30, _ = solve(n=30)
    grid_error = l2_check(c30, a30, norms30)
    quadrature_error = results[30][1]
    agreement = abs(grid_error-quadrature_error)/quadrature_error
    print(f'N=30 L2 check: adaptive quadrature={quadrature_error:.8e}, fine grid={grid_error:.8e}')
    print(f'Relative difference between integration methods = {agreement:.3e}')
    assert np.isclose(grid_error, quadrature_error, rtol=5e-4, atol=1e-10), 'L2 methods disagree'
    print('Independent L2 integration check: PASS')
    report_dir.mkdir(exist_ok=True)
    np.savetxt(report_dir / 'wavefunction_convergence.csv',
               [(n, *results[n]) for n in ns], delimiter=',',
               header='N,relative_energy_error,L2_wavefunction_error,condition_number',
               comments='', fmt=['%d', '%.12e', '%.12e', '%.12e'])
    show()


def solve_pruned(n, alpha=.01, beta=1.3, z=1, cutoff=1e-9):
    """Prune the overlap basis; return (energy in Ha, number kept)."""
    a, norms, S, H = matrices(n, alpha, beta, z)
    values, vectors = eigh(S)
    keep = values > cutoff*values[-1]
    if not np.any(keep):
        return np.nan, 0
    X = vectors[:, keep]/np.sqrt(values[keep])[None, :]

    # Canonical orthogonalisation: discard the smallest overlap directions
    # (see the basis-set discussion in Lehtola et al., 2019).
    # Re-orthogonalise the ones left, since tiny eigenvalues can amplify
    # round-off errors. This was needed to keep the pruning curves reliable.
    gram = X.T @ S @ X
    gram = (gram + gram.T)/2
    gram_values, gram_vectors = eigh(gram)
    if np.any(gram_values <= 0):
        print(f'  N={n}, cutoff={cutoff:.0e}: invalid overlap after pruning')
        return np.nan, int(np.sum(keep))
    X = X @ ((gram_vectors/np.sqrt(gram_values)) @ gram_vectors.T)

    identity_error = np.linalg.norm(X.T @ S @ X - np.eye(np.sum(keep)), ord=np.inf)
    if identity_error > 1e-5:
        print(f'  N={n}, cutoff={cutoff:.0e}: orthogonality error {identity_error:.2e}')
        return np.nan, int(np.sum(keep))

    reduced = X.T @ H @ X
    asymmetry = np.linalg.norm(reduced-reduced.T, ord=np.inf)
    scale = max(1., np.linalg.norm(reduced, ord=np.inf))
    if asymmetry/scale > 1e-6:
        print(f'  N={n}, cutoff={cutoff:.0e}: symmetry error {asymmetry/scale:.2e}')
        return np.nan, int(np.sum(keep))
    reduced = (reduced + reduced.T)/2

    energies, vectors = eigh(reduced, subset_by_index=[0, 0])
    energy, coeff = energies[0], vectors[:, 0]
    residual = np.linalg.norm(reduced @ coeff - energy*coeff)
    residual /= np.linalg.norm(reduced, 2) + abs(energy)
    if residual > 1e-10 or energy < -z*z/2 - 1e-8:
        print(f'  N={n}, cutoff={cutoff:.0e}: eigenpair or variational check failed')
        return np.nan, int(np.sum(keep))
    return energy, int(np.sum(keep))


def plot_basis_pruning():
    """Plot pruning energy errors and the number of basis directions kept."""
    heading('BASIS PRUNING (ADDITIONAL NUMERICAL EXPERIMENT)')
    print('Pruned solutions are checked for numerical reliability.')
    ns = np.arange(4, 41)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for cutoff in (1e-6, 1e-9, 1e-12):
        errors, retained = [], []
        for n in ns:
            energy, used = solve_pruned(int(n), cutoff=cutoff)
            errors.append(relative_error(energy) if used else np.nan)
            retained.append(used)
        print(f'cutoff={cutoff:.0e}: excluded {np.count_nonzero(~np.isfinite(errors))} unstable solutions')
        ax[0].semilogy(ns, errors, '-', lw=2, label=rf'$\tau={cutoff:.0e}$')
        ax[1].plot(ns, retained, lw=2, label=rf'$\tau={cutoff:.0e}$')
    ax[0].set(xlabel='Original number of Gaussians',
              ylabel='Relative energy error', title='(a) Accuracy after pruning')
    ax[1].set(xlabel='Original number of Gaussians',
              ylabel='Retained basis directions', title='(b) Effective basis size')
    ax[0].legend()
    ax[1].legend()
    show()


# Run the complete set of calculations

def main():
    """Run the checks, calculations and plots used in the report."""
    energy, c, a, norms, condition = solve()
    heading('HYDROGENIC GAUSSIAN BASIS REPORT')
    print(f'Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__}')
    print(f'Baseline: N={n_basis}, alpha={alpha_start}, beta={beta_ratio}, Z=1')
    print(f'Gaussian energy       = {energy:.10f} Ha')
    print(f'Exact energy          = {-0.5:.10f} Ha')
    print(f'Relative energy error = {relative_error(energy):.8e}')
    print(f'Gaussian <r>          = {mean_radius(c, a, norms):.8f} bohr')
    print(f'Exact <r>             = {1.5:.8f} bohr')
    print(f'Condition number      = {condition:.6e}')
    print(f'Gaussian psi(0)       = {np.dot(c, norms):.9f} bohr^(-3/2)')
    print(f'Exact psi(0)          = {1/np.sqrt(np.pi):.9f} bohr^(-3/2)')
    print(f'Absolute error in psi(0) = {abs(np.dot(c, norms)-1/np.sqrt(np.pi)):.8e}')
    print('Finite s-Gaussian derivative at r=0: exactly zero')
    print('Exact 1s derivative at r=0:       -1/sqrt(pi)')
    report_dir.mkdir(exist_ok=True)
    with (report_dir / 'hydrogen_benchmark.csv').open('w') as file:
        file.write('quantity,value,units\n')
        for name, value, unit in [
            ('N', n_basis, 'count'), ('alpha', alpha_start, 'bohr^-2'),
            ('beta', beta_ratio, 'dimensionless'), ('gaussian_energy', energy, 'Ha'),
            ('exact_energy', -.5, 'Ha'),
            ('relative_energy_error', relative_error(energy), 'dimensionless'),
            ('gaussian_mean_radius', mean_radius(c,a,norms), 'bohr'),
            ('exact_mean_radius', 1.5, 'bohr'),
            ('condition_number', condition, 'dimensionless'),
            ('gaussian_psi_at_zero', np.dot(c,norms), 'bohr^-3/2'),
            ('exact_psi_at_zero', 1/np.sqrt(np.pi), 'bohr^-3/2')]:
            file.write(f'{name},{value:.14g},{unit}\n')

    validate()
    basis_selection()

    heading('SAVING ALL TWELVE FIGURES')
    print(f'Project directory: {Path(__file__).resolve().parent}')
    print(f'Figure directory: {figure_dir}')
    plot_overview()
    plot_hydrogen(energy, c, a, norms)
    plot_convergence()
    plot_parameter_map()
    plot_charge_transfer()
    plot_scaling_collapse()
    plot_nuclear_cusp(c, a, norms)
    plot_local_energy(c, a, norms)
    plot_wavefunction_convergence()
    plot_basis_pruning()
    assert figure_number == len(figure_names), 'Not all report figures were generated'
    heading('FINISHED')
    print(f'All {figure_number} figures saved to: {figure_dir}')
    print('Saved figures:')
    for name in saved_figures:
        print('  ' + name)
    print(f'Numerical tables: {report_dir}')
    print('All validation checks passed.')


if __name__ == '__main__':
    main()
