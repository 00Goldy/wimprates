"""
Migdal effect

Two implemented models:
 * Ibe et al: https://arxiv.org/abs/1707.07258
 * Cox et al: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.035032

 In the energy range of DM, the dipole approximation model implemented by Ibe et al
 is compatible with the one developped by Cox et al (check discussion in Cox et al)
"""

from collections.abc import Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, Optional, Union

from fnmatch import fnmatch
from functools import lru_cache, partial
import numericalunits as nu
import numpy as np
import pickle as pkl
from tqdm.autonotebook import tqdm
import pandas as pd
from scipy.integrate import quad
from scipy.integrate import trapezoid
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import wimprates as wr


export, __all__ = wr.exporter()


@dataclass
class Shell:
    """
    Describes a specific atomic shell for the selected atom.

    Attributes:
        name (str): The name of the shell.
        element (str): The element class of the atom.
        binding_e (float): The binding energy for the shell.
        model (str): The model used for the single ionization probability computation.
        single_ionization_probability (Callable): A function to assign interpolators to.
            The interpolator will provide the single ionization probability for the shell
            according to the selected model.

    Methods:
        __call__(*args, **kwargs) -> np.ndarray:
            Calls the single_ionization_probability function with the given arguments and keyword arguments.

    Properties:
        n (int): Primary quantum number.
        l (str): Azimuthal quantum number for Ibe; Azimuthal + magnetic quantum number for Cox.
    """

    name: str
    element: str
    binding_e: float
    model: str
    single_ionization_probability: Callable  # to assign interpolators to

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.single_ionization_probability(*args, **kwargs)

    @property
    def n(self) -> int:
        return int(self.name[0])

    @property
    def l(self) -> str:
        return self.name[1:]


def _default_shells(material: str) -> tuple[str]:
    """
    Returns the default shells to consider for a given material.
    Args:
        material (str): The material for which to determine the default shells.
    Returns:
        list[str]: The default shells to consider for the given material.
    """

    consider_shells = dict(
        # For Xe, only consider n=3 and n=4
        # n=5 is the valence band so unreliable in liquid
        # n=1,2 contribute very little
        Xe=["3*", "4*"],
        # TODO, what are realistic values for Ar?
        Ar=["2*"],
        # EDELWEIS
        Ge=["3*"],
        Si=["2*"],
    )
    return tuple(consider_shells[material])


def _create_cox_probability_function(
    element,
    orbital: str,
    dipole: bool = False,
) -> Callable[..., np.ndarray[Any, Any]]:

    fn_name = "dpI1dipole" if dipole else "dpI1"
    fn = getattr(element, fn_name)

    return partial(fn, orbital=orbital)


@export
def get_migdal_transitions_probability_iterators(
    material: str = "Xe",
    model: str = "Ibe",
    considered_shells: Optional[Union[tuple[str], str]] = None,
    dark_matter: bool = True,
    e_threshold: Optional[float] = None,
    dipole: bool = False,
    **kwargs,
) -> list[Shell]:
    # Differential transition probabilities for <material> vs energy (eV)

    # Check if considered_shells is an empty list
    if considered_shells is None:
        considered_shells = _default_shells(material)

    shells = []
    if model == "Ibe":
        df_migdal_material = pd.read_csv(
            wr.data_file("migdal/Ibe/migdal_transition_%s.csv" % material)
        )

        # Relevant (n, l) electronic states
        migdal_states_material = df_migdal_material.columns.values.tolist()
        migdal_states_material.remove("E")

        # Binding energies of the relevant electronic states
        # From table II of 1707.07258
        energy_nl = dict(
            Xe=np.array(
                [
                    3.5e4,
                    5.4e3,
                    4.9e3,
                    1.1e3,
                    9.3e2,
                    6.6e2,
                    2.0e2,
                    1.4e2,
                    6.1e1,
                    2.1e1,
                    9.8,
                ],
            ),
            Ar=np.array([3.2e3, 3.0e2, 2.4e2, 2.7e1, 1.3e1]),
            Ge=np.array([1.1e4, 1.4e3, 1.2e3, 1.7e2, 1.2e2, 3.5e1, 1.5e1, 6.5e0]),
            # http://www.chembio.uoguelph.ca/educmat/atomdata/bindener/grp14num.htm
            Si=np.array([1844.1, 154.04, 103.71, 13.46, 8.1517]),
        )

        for state, binding_e in zip(migdal_states_material, energy_nl[material]):
            if not any(fnmatch(state, take) for take in considered_shells):
                continue
            binding_e *= nu.eV

            # Lookup for differential probability (units of ev^-1)
            p = interp1d(
                np.array(df_migdal_material["E"].values) * nu.eV,
                df_migdal_material[state].values / nu.eV,
                bounds_error=False,
                fill_value=0,
            )
            #print(p(2*nu.eV)*nu.eV)
            #print('E :'+str(np.array(df_migdal_material["E"].values)))
            #print('P :'+str(df_migdal_material[state].values))

            shells.append(Shell(state, material, binding_e, model, p))

    elif model == "Cox":
        element = wr.cox_migdal_model(
            material,
            dipole=dipole,
            dark_matter=dark_matter,
            e_threshold=e_threshold,
            **kwargs
        )

        for state, binding_e in element.orbitals:
            if not any(fnmatch(state, take) for take in considered_shells):
                continue

            shells.append(
                Shell(
                    state,
                    material,
                    binding_e * nu.keV,
                    model,
                    single_ionization_probability=_create_cox_probability_function(
                        element,
                        state,
                        dipole=dipole,
                    ),
                )
            )
    else:
        raise ValueError("Only 'Cox' and 'Ibe' models have been implemented")

    return shells


def vmin_migdal(
    w: np.ndarray, erec: np.ndarray, mw: float, material: str
) -> np.ndarray:
    """Return minimum WIMP velocity to make a Migdal signal with energy w,
    given elastic recoil energy erec and WIMP mass mw.
    """
    y = (wr.mn(material) * erec / (2 * wr.mu_nucleus(mw, material) ** 2)) ** 0.5
    y += w / (2 * wr.mn(material) * erec) ** 0.5
    return np.maximum(0, y)


def get_diff_rate(
    w: float,
    shells: list[Shell],
    mw: float,
    sigma_nucleon: float,
    halo_model: wr.StandardHaloModel,
    interaction: str,
    m_med: float,
    migdal_model: str,
    include_approx_nr: bool,
    q_nr: float,
    material: str,
    t: Optional[float],
    **kwargs,
):
    result = 0
    for shell in shells:

        def diff_rate(v, erec):
            # Observed energy = energy of emitted electron
            #                 + binding energy of state
            eelec = w - shell.binding_e - include_approx_nr * erec * q_nr
            if eelec < 0:
                return 0

            if migdal_model == "Ibe":
                return (
                    # Usual elastic differential rate,
                    # common constants follow at end
                    wr.sigma_erec(
                        erec,
                        v,
                        mw,
                        sigma_nucleon,
                        interaction,
                        m_med=m_med,
                        material=material,
                    )
                    * v
                    * halo_model.velocity_dist(v, t)
                    # Migdal effect |Z|^2
                    # TODO: ?? what is explicit (eV/c)**2 doing here?
                    * (nu.me * (2 * erec / wr.mn(material)) ** 0.5 / (nu.eV / nu.c0))
                    ** 2
                    / (2 * np.pi)
                    * shell(eelec)
                )
            elif migdal_model == "Cox":
                vrec = (2 * erec / wr.mn(material)) ** 0.5 / nu.c0
                input_points = wr.pairwise_log_transform(eelec/nu.keV, vrec)
                return (
                    wr.sigma_erec(
                        erec,
                        v,
                        mw,
                        sigma_nucleon,
                        interaction,
                        m_med=m_med,
                        material=material,
                    )
                    * v
                    * halo_model.velocity_dist(v, t)
                    * shell(input_points) / nu.keV
                )

        # Note dblquad expects the function to be f(y, x), not f(x, y)...
        result += dblquad(
            diff_rate,
            0,
            wr.e_max(mw, wr.v_max(t, halo_model.v_esc), wr.mn(material)),
            lambda erec: vmin_migdal(
                w=w - include_approx_nr * erec * q_nr,
                erec=erec,
                mw=mw,
                material=material,
            ),
            lambda _: wr.v_max(t, halo_model.v_esc),
            **kwargs,
        )[0]

    return result


@export
def rate_migdal(
    w: Union[np.ndarray, float],
    mw: float,
    sigma_nucleon: float,
    interaction: str = "SI",
    m_med: float = float("inf"),
    include_approx_nr: bool = False,
    q_nr: float = 0.15,
    material: str = "Xe",
    t: Optional[float] = None,
    halo_model: Optional[wr.StandardHaloModel] = None,
    consider_shells: Optional[tuple[str]] = None,
    migdal_model: str = "Ibe",
    dark_matter: bool = True,
    dipole: bool = False,
    e_threshold: Optional[float] = None,
    progress_bar: bool = False,
    multi_processing: Optional[Union[bool, int]] = True,
    **kwargs,
) -> np.ndarray:
    """Differential rate per unit detector mass and deposited ER energy of
    Migdal effect WIMP-nucleus scattering

    :param w: ER energy deposited in detector through Migdal effect
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    See sigma_erec for options
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param include_approx_nr: If True, instead return differential rate
        per *detected* energy, including the contribution of
        the simultaneous NR signal approximately, assuming q_{NR} = 0.15.
        This is how https://arxiv.org/abs/1707.07258
        presented the Migdal spectra.
    :param q_nr: conv between Enr and Eee (see p. 27 of
        https://arxiv.org/pdf/1707.07258.pdf)
    :param material: name of the detection material (default is 'Xe')
    :param t: A J2000.0 timestamp.
    If not given, conservative velocity distribution is used.
    :param halo_model: class (default to standard halo model)
    containing velocity distribution
    :param consider_shells: consider the following atomic shells, are
        fnmatched to the format from Ibe (i.e. 1_0, 1_1, etc).
    :param progress_bar: if True, show a progress bar during evaluation
    (if w is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    _is_array = True
    if not isinstance(w, np.ndarray):
        if isinstance(w, float):
            _is_array = False
            w = np.array([w])
        else:
            raise ValueError("w must be a float or a numpy array")

    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model

    if progress_bar:
        prog_bar = tqdm
    else:
        prog_bar = lambda x, *args, **kwargs: x

    if not consider_shells:
        consider_shells = _default_shells(material)

    shells = get_migdal_transitions_probability_iterators(
        material=material,
        model=migdal_model,
        considered_shells=consider_shells,
        dipole=dipole,
        e_threshold=e_threshold,
        dark_matter=dark_matter,
    )

    if multi_processing and not dipole:
        multi_processing = None if isinstance(multi_processing, bool) else multi_processing
        with ProcessPoolExecutor(multi_processing) as executor:
            partial_get_diff_rate = partial(
                get_diff_rate,
                shells=shells,
                mw=mw,
                sigma_nucleon=sigma_nucleon,
                halo_model=halo_model,
                interaction=interaction,
                m_med=m_med,
                migdal_model=migdal_model,
                include_approx_nr=include_approx_nr,
                q_nr=q_nr,
                material=material,
                t=t,
            )

            n_workers = os.cpu_count() if multi_processing is None else multi_processing
            results = list(
                prog_bar(
                    executor.map(partial_get_diff_rate, w),
                    desc=f"Computing rates (MP={n_workers} workers)",
                    total=len(w),
                )
            )
    else:
        results = []
        for val in prog_bar(w, desc="Computing rates"):
            results.append(
                get_diff_rate(
                    val,
                    shells,
                    mw,
                    sigma_nucleon,
                    halo_model,
                    interaction,
                    m_med,
                    migdal_model,
                    include_approx_nr,
                    q_nr,
                    material,
                    t,
                )
            )

    results = np.array(results) if _is_array else float(results[0])
    return halo_model.rho_dm / mw * (1 / wr.mn(material)) * results

# New function for Migdal from CEvNS
#######################################################################################################################################
@export
def rate_migdal_cevns(
    E_e: np.ndarray,
    flux_nu: Callable[[float], float],
    dsigma: Callable[[float, float], float],
    q_nr: float = 0.15,
    material: str ='Xe',
    dark_matter: bool = True,
    dipole: bool = False,
    migdal_model: str = 'Ibe',
    consider_shells: Optional[tuple[str]] = ["1*","2*","3*","4*","5*"],
    E_nu_min: float = 0.6,  # MeV 
    E_nu_max: float = 20, # MeV 
    include_approx_nr: bool = True,
    **kwargs
)-> np.ndarray:
    """Differential rate per unit detector mass and deposited ER energy of
    Migdal effect WIMP-nucleus scattering

    :param E_e: ER energy deposited in detector through Migdal effect 
    :param flux_nu: Neutrino flux received by the detector (cm^-2.s^-1.MeV^-1) 
    :param dsigma: neutrino/nucleon cross-section for given interaction
    :param consider_shells: consider the following atomic shells, are
        fnmatched to the format from Ibe (i.e. 1_0, 1_1, etc).
    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    m_N = wr.mn(material) # We get the mass of a Xe nucleus (in amu)
    m_N *= 931.5e6*nu.eV/nu.amu # We convert it to eV/c^2 so eV in natural units
    Result = []
    conv = 3.154e7 * nu.NA / (m_N / nu.amu * 1e-6) # Conversion from kg^-1 . s^-1 -> tons^-1 . yr^-1

    # Condition for the shells considered
    if consider_shells is None:
        consider_shells = _default_shells(material)
    
    # We get the range of energy where the probability is defined
    # ~~~~~~~~~~~~~~~~~~~
    df_migdal_material = pd.read_csv(
            wr.data_file("migdal/Ibe/migdal_transition_%s.csv" % material)
        )

    E_e_data = df_migdal_material['E'].copy() #In eV
    # ~~~~~~~~~~~~~~~~~~~

    def E_rmax(E_nu): 
        """ Highest NR recoil energy transfered by neutrino of E_nu energy
        :param E_nu: Neutrino energy in eV
        :result E_rmax: NR recoil energy in eV
        """
        return 2 * (E_nu * nu.eV)**2 / (m_N + 2*E_nu * nu.eV) * 1/nu.eV
    
    # We define all the shells for our material and the ionization probability associated
    shells = get_migdal_transitions_probability_iterators(
        material=material,
        model=migdal_model,
        considered_shells=consider_shells,
        dipole=dipole,
        dark_matter=dark_matter,
    )

    def p_diff_func(E_nr, E_det, shell):
        """ Differential probability for a given shell and nucleus momentum
        :param E_nr: NR energy in eV
        :param E_det: Detcted energy in eV
        :param shell: A callable which return the single inonization differential probability (in nu.eV^-1) for a given energy (in nu.eV)

        :result p_diff_func: Single ionization differential probability in eV^-1
        """
        return shell(E_det*nu.eV)*nu.eV /(2*np.pi) * (nu.me  * (2 * E_nr * nu.eV /m_N) ** 0.5 / (nu.eV / nu.c0**2))** 2


    ### First test: Plotting differential probability for Xenon, summed by principal quantum number n
    """
    # Fixed nuclear recoil energy for a nucleus velocity of 10^-3 in units of c
    E_nr_fixed = 1e-6 * m_N / (2 * nu.eV) # eV

    # Electronic recoil energies in keV (log-spaced)
    E_e = np.logspace(0, 4 + np.log10(2), 1000)

    # Group shells by principal quantum number n (based on shell.name like '3_2')
    shells_by_n = defaultdict(list)
    for shell in shells:
        n = int(shell.name.split('_')[0])  # Extract principal quantum number
        shells_by_n[n].append(shell)

    # Compute and sum differential probabilities for each group of shells with the same n
    p_diff_grouped = {}
    for n, shell_group in shells_by_n.items():
        p_diff_sum = np.zeros_like(E_e)
        for shell in shell_group:
            p_diff_shell = [p_diff_func(E_nr_fixed, E_er_value, shell) for E_er_value in E_e]
            p_diff_sum += np.array(p_diff_shell)
        p_diff_grouped[n] = p_diff_sum

    # Plotting
    plt.figure(figsize=(8, 6))
    for n, p_diff in sorted(p_diff_grouped.items()):
        plt.plot(E_e, p_diff, label=f'n={n}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E_e$ [eV]', fontsize=16)
    plt.ylabel(r'$\frac{1}{2\pi} \frac{dp^c}{dE_e}$ [eV$^{-1}$]', fontsize=16)
    plt.title(r'Migdal Effect: $\frac{dp^c}{dE_e}$ summed over $l$ for each $n$', fontsize=14)
    plt.legend(title="Principal quantum number")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
    """
    

    ### Second test: Computing total probabilities over E_e
    """
    # Fixed nuclear recoil energy for a nucleus velocity of 10^-3 in units of c
    E_nr_fixed = 1e-6 * m_N / (2 * nu.eV)

    for shell in shells:
        p_diff_shell = []
        for E_er_value in E_e_data : 
            p_diff_shell.append(p_diff_func(E_nr_fixed, E_er_value, shell))
        Pshell = trapezoid(p_diff_shell, E_e_data)

        print(str(shell.name) + " : " + str(Pshell))
    """


    # We define the original differential rate to compare
    def rate_no_migdal(E_nu, E_nr):
        """ Differential Rate for a given neutrino Energy and NR energy
        :param E_nu: Neutrino energy in eV
        :param E_nr: NR energy in eV
        
        :result rate_no_migdal: Differential probability without computing the migdal probability
        """
        return flux_nu(E_nu/(nu.MeV/nu.eV))/(nu.MeV/nu.eV) * dsigma(E_nu/(nu.MeV/nu.eV), E_nr/(nu.MeV/nu.eV))
    
    def integrand_diff(E_nu, E_nr, E_det):
        """ Computing dR_mig / (dE_nu . dE_nr . dE_det) to later be integrated over E_nu and E_nr
        :param E_nu: Neutrino energy in eV
        :param E_nr: NR energy in eV
        
        :result integrand_diff: integrand to be integrated
        """
        #return flux_nu(E_nu/(nu.MeV/nu.eV))/(nu.MeV/nu.eV) * dsigma(E_nu/(nu.MeV/nu.eV), E_nr/((nu.MeV/nu.eV))) * P_Mig
        return flux_nu(E_nu/(nu.MeV/nu.eV)) * dsigma(E_nu/(nu.MeV/nu.eV),E_nr/(nu.MeV/nu.eV)) * p_diff_func(E_nr, E_det, shell), p_diff_func(E_nr, E_det, shell) 
    """
    df = pd.DataFrame({
        "energy_keV": ENR,
        "spectrum_value_norm": Result
        })

    df.to_pickle("Migdal_CEvNS_solar_spectrum.pkl")
    """
    rate = {}

    for shell in shells :
        E = []
        res_shell = []

        for E_er in E_e :

            def integrand_wrapped(E_nu, E_nr):
                E_det_value = E_er - shell.binding_e /nu.eV - include_approx_nr * E_nr * nu.MeV / nu.eV *q_nr
                if E_det_value <= 0:
                    return 0.0
                val, p = integrand_diff(E_nu * nu.MeV / nu.eV, E_nr * nu.MeV / nu.eV, E_det_value)
                E.append(E_det_value)
                return val

            res_shell_value, err = dblquad(
                integrand_wrapped,
                E_nu_min * nu.MeV / nu.eV, E_nu_max * nu.MeV / nu.eV,       # E_nu (eV)
                lambda E_nu: 0,                                             # E_nr min (eV)
                lambda E_nu: E_rmax(E_nu * nu.MeV / nu.eV),                 # E_nr max (eV)
                **kwargs
            )

            res_shell.append(res_shell_value)
        rate[shell.name] = interp1d(E, res_shell, bounds_error=False, fill_value=0)
    
    E_arb = np.linspace(0, 3.5e4 + E_e[len(E_e)-1]) # We define an arbitrary range of energy 
    for interp in rate.values():
        total_vals += interp(E_arb)
    # Créer l'interpolateur final
    rate_total = interp1d(E_arb, total_vals, bounds_error=False, fill_value=0)

    return rate_total




#######################################################################################################################################

@wr.deprecated("Use get_migdal_transitions_probability_iterators instead")
@lru_cache()
def read_migdal_transitions(material="Xe"):
    ### (DEPRECATED) Maintain this for backwards accessibility
    # Differential transition probabilities for <material> vs energy (eV)

    df_migdal_material = pd.read_csv(
        wr.data_file("migdal/Ibe/migdal_transition_%s.csv" % material)
    )

    # Relevant (n, l) electronic states
    migdal_states_material = df_migdal_material.columns.values.tolist()
    migdal_states_material.remove("E")

    # Binding energies of the relevant electronic states
    # From table II of 1707.07258
    energy_nl = dict(
        Xe=np.array(
            [3.5e4, 5.4e3, 4.9e3, 1.1e3, 9.3e2, 6.6e2, 2.0e2, 1.4e2, 6.1e1, 2.1e1, 9.8]
        ),
        Ar=np.array([3.2e3, 3.0e2, 2.4e2, 2.7e1, 1.3e1]),
        Ge=np.array([1.1e4, 1.4e3, 1.2e3, 1.7e2, 1.2e2, 3.5e1, 1.5e1, 6.5e0]),
        # http://www.chembio.uoguelph.ca/educmat/atomdata/bindener/grp14num.htm
        Si=np.array([1844.1, 154.04, 103.71, 13.46, 8.1517]),
    )

    binding_es_for_migdal_material = dict(
        zip(migdal_states_material, energy_nl[material])
    )

    return (
        df_migdal_material,
        binding_es_for_migdal_material,
    )
