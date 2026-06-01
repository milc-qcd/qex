import os
import json
import datetime
import statistics as stat

import scipy
import numpy as np
import streamlit as st
import pyerrors as pe
import gvar as gv

#import emcee

import matplotlib.pyplot as plt

import plot

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data') + os.sep

### information ###

DATASETS = {
    '20': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000'],
        '750': ['000', '0005', '00025', '0001'],
        '725': ['0005', '00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000']
    },
    '24': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000'],
        '750': ['000', '0005', '00025', '0001'],
        '725': ['0005', '00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000']
    },
    '32': {
        '200': ['000'],
	'180': ['000'],
	'160': ['000'],
	'140': ['000'],
	'120': ['000'],
        '110': ['000'],
	'100': ['000'],
        '950': ['000'],
        '900': ['000'],
	'850': ['000'],
	'800': ['000'],
	'750': ['000', '0005', '00025', '0001'],
        '725': ['0005',	'00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000']
    },
    '40': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '110': ['000'],
        '100': ['000'],
        '950': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000'],
	'750': ['000', '0005', '00025', '0001'],
        '725': ['0005',	'00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000']
    },
    '48': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '110': ['000'],
        '100': ['000'],
        '950': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000'],
        '750': ['000', '0005', '00025', '0001'],
        '725': ['0005', '00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000']
    },
    '64': {
        '160': ['000']
    }
}

INTEGRATOR = {'2MN': '2nd-order Omelyan'}
VOLUMES = {
    "20": r"$20^3 \times 40$",
    "24": r"$24^3 \times 48$",
    "32": r"$32^3 \times 64$",
    "40": r"$40^3 \times 80$",
    "48": r"$48^3 \times 96$",
    "64": r"$64^3 \times 128$",
    "all": r"all"
}
BETAS = {
    "675": "6.75",
    "700": "7.00",
    "725": "7.25",
    "750": "7.50",
    "800": "8.00",
    "850": "8.50",
    "900": "9.00",
    "950": "9.50",
    "100": "10.0",
    "110": "11.0",
    "120": "12.0",
    "140": "14.0",
    "160": "16.0",
    "180": "18.0",
    "200": "20.0"
}
MASSES = {
    "000"  : "0.0000",
    "0005" : "0.0050",
    "00025": "0.0025",
    "0001" : "0.0010"
}

ENSVOL = {
    "20": "l20t40",
    "24": "l24t48",
    "32": "l32t64",
    "40": "l40t80",
    "48": "l48t96",
    "64": "l64t128"
}

### helper procedures ###

def weighted(data, expdh):
    obs = pe.Obs([data], ['dummy'])
    obs.gamma_method()
    #return (
    #    gv.gvar(obs.e_tauint['dummy'], obs.e_dtauint['dummy']), 
    #    gv.gvar(obs.value, obs.dvalue)
    #)
    gvobs = gv.dataset.avg_data(data)
    return (data, gv.gvar(obs.value, obs.dvalue))
    #obs = pe.Obs([data], ['dummy'])
    #obs.gamma_method()
    #tau = obs.e_tauint['dummy']
    #try:
    #    p = [1 for v in expdh]
    #    p = np.array(p)/sum(p)
    #    m = sum(data*p)
    #    e = np.sqrt(tau*sum((data - m)*(data - m)*p))
    #    return (data*expdh, gv.gvar(m, e))
    #except ValueError:
    #    (tdata, texpdh) = (list(data), list(expdh))
    #    if len(tdata) > len(texpdh):
    #        while len(tdata) != len(texpdh): del tdata[-1]
    #    elif len(data) < len(expdh):
    #        while len(tdata) != len(texpdh): del texpdh[-1]
    #    (tdata, texpdh) = (np.array(tdata), np.array(texpdh))
    #    tp = [1 for v in texpdh]
    #    tp = np.array(tp)/sum(tp)
    #    m = sum(tdata*tp)
    #    e = np.sqrt(tau*sum((tdata - m)*(tdata - m)*tp))
    #    return (tdata*texpdh, gv.gvar(m, e))

def proper(data):
    obs = pe.Obs([data], ['dummy'])
    obs.gamma_method()
    return (
        gv.gvar(obs.e_tauint['dummy'], obs.e_dtauint['dummy']), 
        gv.gvar(obs.value, obs.dvalue)
    )

### page layout ###

# page configuration
st.set_page_config(page_title = 'ensembles', page_icon = ':atom_symbol:')

# page title
st.markdown(r"# Fermilab Lattice and MILC Collaboration $\boldsymbol{\alpha_{\mathrm{s}}(M_{Z})}$ ensembles")

# navigation information
st.markdown("""
""")

# Construct side bar
st.sidebar.markdown("""
## Ensemble Parameters
""")

### side bar ###

# volume selector
volume = st.sidebar.selectbox(
    r"$N_{\mathrm{s}} \equiv L/a$", 
    options = [*VOLUMES.keys()]
)

coupling: float = 0.0
mass: float = 0.0
ensemble: str = ''
fn: str = ''

if volume != "all":
    # bare gauge coupling selector
    coupling = st.sidebar.selectbox(
        r"$\beta_{b} \equiv 6/g_{0}^2$", 
        options = [*DATASETS[volume].keys()][::-1],
        format_func = lambda beta: BETAS[beta]
    )

    # bare gauge coupling selector
    mass = st.sidebar.selectbox(
        r"$am_{\mathrm{f}}$", 
        options = DATASETS[volume][coupling][::-1],
        format_func = lambda m: MASSES[m]
    )

    # action type toggle
    action_type = st.sidebar.radio(
        "Action type",
        ["HISQ", "HYP"],
        horizontal = True,
        key = f"action_{volume}_{coupling}_{mass}"
    )

    ### display ensemble information ###

    ensemble = ''.join(['f4', ENSVOL[volume], 'b', coupling, 'm', mass, f'_{action_type}_pppa'])
    fn = ensemble + '-info.json'

if os.path.exists(DATA_DIR + fn) and volume != 'all':
    ### side bar display ###
    with open(DATA_DIR + fn, 'r') as in_file: data = json.load(in_file)
    st.sidebar.markdown(
    f"""
    ## Summary \n
    - configurations: {len(data['dH']) - sum(data['cut'])}
    - running: {data['running']}
    - last update: {datetime.datetime.fromtimestamp(os.path.getctime(DATA_DIR + fn))}
    
    ## Molecular Dynamics
    - trajectory length: {data['hmc']['trajectory-length']}
    - outer integrator 
      - fields: fermion and Hasenbusch
      - algorithm: {INTEGRATOR[data['fermion']['integrator']]}
      - steps: {data['fermion']['steps']}
    - inner integrator
      - fields: gauge
      - algorithm: {INTEGRATOR[data['gauge']['integrator']]}
      - steps: {data['gauge']['steps']}*

    ## Action
    - Hasenbusch mass: {data['action']['hasenbusch-mass']}
    - plaquette coefficient: 1
    - rectangle coefficent: -1/20
    - tadpole: 1
    - fat7 (level 1)
      - one link: 1/8
      - three link: 1/16
      - five link: 1/64
      - seven link: 1/384
      - lepage: 0.0
    - fat7 (level 2)
      - one link: 1
      - three link: 1/16
      - five link: 1/64
      - seven link: 1/384
      - lepage: -1/8
    - naik: -1/24
    - unitary projection: Cayley-Hamilton

    --------

    *per outer integrator gauge field update 
    """)

    ### main page plots ###

    cfgs = [*range(len(data['dH']))]

    splaq = np.array(data['spatial plaquette'])
    tplaq = np.array(data['temporal plaquette'])

    # fcn(dH)
    dH = np.array(data['dH']) #np.array([d for idx,d in enumerate(data['dH']) if not data['cut'][idx]])
    dH_cut = [d for idx,d in enumerate(data['dH']) if not data['cut'][idx]]
    #dH = np.array([dh if data['acceptance'][c] else 0.0 for c,dh in enumerate(data['dH'])])
    dH2 = dH*dH
    expdH = np.exp(-dH)
    expdH_cut = [d for idx,d in enumerate(expdH) if not data['cut'][idx]]
    dH2_cut = [d for idx,d in enumerate(dH2) if not data['cut'][idx]]
    pred_acc_rate = int(round(scipy.special.erfc(np.sqrt(np.mean(dH2_cut)/8.))*100.))
    acc_rate = [min(1.0, expdh) for expdh in expdH_cut]
    avg_acc_rate = int(round(np.mean(acc_rate)*100.))
    
    # cg iterations
    hcg = data['average CG iterations (hasenbusch)']
    fcg = data['average CG iterations (fermion)']

    def plot_hmc_observable(h, meas, lbl, bins = None, range = None):
        cfg_cut = [cfg for idx, cfg in enumerate(cfgs) if not data['cut'][idx]] 
        meas_cut = [m for idx, m in enumerate(meas) if not data['cut'][idx]]
        if bins is None: bins = len(meas_cut)

        # plot data before cut
        h.scatter(
            [cfg for idx, cfg in enumerate(cfgs) if data['cut'][idx]], 
            [m for idx, m in enumerate(meas) if data['cut'][idx]], 
            **{'facecolor': 'none', 'edgecolor': 'grey', 'alpha': 0.1}
        )
        
        # create vertical line indicating location of cut
        h.axis.axvline(min(cfg_cut), color = 'k')

        # create horizontal line indicating mean and error
        expdh_cut = [m for idx, m in enumerate(expdH) if not data['cut'][idx]]
        (meas_cut, expdh_cut) = (np.array(meas_cut), np.array(expdh_cut))
        #(w, exp) = weighted(meas_cut, expdh_cut)
        (tau, exp) = proper(meas_cut)
        x = [min(cfg_cut), max(cfg_cut) - 1]
        y = [exp, exp]
        h.fill_between(x, y, color = 'magenta', alpha = 0.1)
        h.axis.axhline(exp.mean, color = 'magenta', alpha = 0.5)

        # plot data after cut
        try:
            h.scatter(
                cfg_cut, meas_cut, 
                **{'facecolor': 'none', 'edgecolor': 'grey', 'alpha': 0.5}
            )
        except ValueError:
            # correct for mismatch in configuration count
            if len(cfg_cut) > len(meas_cut):
                while len(cfg_cut) != len(meas_cut): del cfg_cut[-1]
            elif len(cfg_cut) < len(meas_cut):
                while len(cfg_cut) != len(meas_cut): 
                    del dH[-1]
                    del dH2[-1]
                    del expdH[-1]
            h.scatter(
                cfg_cut, meas_cut, 
                **{'facecolor': 'none', 'edgecolor': 'grey', 'alpha': 0.5}
            )

        # attach two histograms to side
        histargs = {
            'orientation': 'horizontal', 
            'color': 'magenta', 
            'alpha': 0.4,
            'edgecolor': 'magenta',
            'linewidth': 1.2
        }
        if range is not None: histargs['range'] = range
        ax_hist = h.attach_histogram(meas_cut, bins, **histargs)
        ax_hist.tick_params(axis = 'y', labelleft = False)
        #histargs['color'] = 'grey'
        #histargs['edgecolor'] = 'grey'
        #ax_hist.hist(meas_cut, bins, **histargs)
        ax_hist.tick_params(
            top = True, 
            labeltop = True, 
            bottom = True, 
            labelbottom = False
        )
        h.set_histogram_title('$\\mathrm{count}$')

        # extra
        ttl = '$\\overline{' + lbl + '} =' + str(exp) + '$, '
        ttl += '$\\tau = ' + str(tau) + '$'
        h.set_title(ttl) 
    
    # information
    st.markdown("""
    ## Hamiltonian Monte Carlo metrics
    """)

    # cumulative acceptance rate
    st.markdown("""
    ### Acceptance
    """)
    dhp = plot.Plot(h = 2.25)
    allacc = [1 for idx, _ in enumerate(data['acceptance']) if not data['cut'][idx]]
    accs = [m for idx, m in enumerate(data['acceptance']) if not data['cut'][idx]]
    cfg_cut = [cfg for idx, cfg in enumerate(cfgs) if not data['cut'][idx]] 
    nrm = np.cumsum(allacc)
    accr = np.cumsum(accs)
    cum_acc_rate = accr/nrm
    dhp.line(cfg_cut, cum_acc_rate , color = 'grey')
    dhp.axis.axvline(min(cfg_cut), color = 'k')
    dhp.set_title(
        '$\\mathrm{acceptance \\ rate \\ = \\ ' + str(cum_acc_rate[-1]) + '}$'
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        ylim = [0., 1.],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\mathrm{cumulative \\ acc \\ rate}$'
    )
    dhp.axis.axhline(cum_acc_rate[-1], color = 'magenta', alpha = 0.5)
    st.pyplot(
        fig = dhp.handle,
        clear_figure = True    
    )
    st.markdown("""
    Cumulative acceptance rate is indicated by grey curve. Last value of cumulative
    acceptance rate is quoted as the acceptance rate. From Creutz's equality (see 
    below), one has for the predicted acceptance rate (on large volumes)
    """)
    st.latex("""
    P \\approx \mathrm{erfc}\\bigg[\\frac{1}{8}\\Big\\langle \mathrm{d}\mathcal{H}^2 \\Big\\rangle^{1/2} \\bigg]
    """)
    st.markdown(f"yielding $P \\approx {pred_acc_rate}\%$ for the present simulation. We also have for the average acceptance rate")
    st.latex("""
    P = \langle \mathrm{min}(1, \exp(-\mathrm{d}\mathcal{H})) \\rangle,
    """)
    st.markdown(f"yielding $P \\approx {avg_acc_rate}\%$")

    # dH
    dhp = plot.Plot(h = 2.25)
    plot_hmc_observable(
        dhp, dH, '\mathrm{d}\mathcal{H}', 
        bins = int(round(0.0025*len(dH))), 
        range = (-1., 1.)
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        #ylim = [-1., 1.],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\mathrm{d}\mathcal{H}$'
    )
    dhp.axis.axhline(0.0, color = 'k', alpha = 0.25)
    st.pyplot(fig = dhp.handle)

    # dH^2
    dhp = plot.Plot(h = 2.25)
    plot_hmc_observable(
        dhp, dH2, '\mathrm{d}\mathcal{H}^2',
        bins = int(round(0.0025*len(dH2))), 
        range = (0., 2.)
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        ylim = [-0.1, 2.0],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\mathrm{d}\mathcal{H}^2$'
    )
    st.pyplot(fig = dhp.handle)

    # exp(-dH)
    #act = emcee.autocorr.integrated_time(expdH_cut, quiet = True)
    #print("AUTOCORRELATION: ", act)
    dhp = plot.Plot(h = 2.25)
    plot_hmc_observable(
        dhp, expdH, '\exp(-\mathrm{d}\mathcal{H})',
        bins = int(round(0.0025*len(expdH))), 
        range = (0., 2.)
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        ylim = [0.0, 2.0],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\exp(-\mathrm{d}\mathcal{H})$'
    )
    dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
    st.pyplot(fig = dhp.handle)
    st.markdown("""
    In the infinite-statistics limit,
    """)
    st.latex("\langle \exp(-\mathrm{d}\mathcal{H}) \\rangle = 1,")
    expdH_exp = gv.gvar(np.mean(expdH_cut), np.std(expdH_cut, ddof = 1)/np.sqrt(len(expdH_cut)))
    nboot = 5000
    size = expdH_exp
    expdH_bstrap = scipy.stats.bootstrap(
        (expdH_cut[0::],), np.mean, confidence_level = 0.68
    ).bootstrap_distribution
    expdH_exp_bstrap = gv.dataset.avg_data(expdH_bstrap, bstrap = True)
    st.markdown(f"""
    sometimes referred to as Creutz's equality. We get: {expdH_exp_bstrap}.
    """)
    #expdH_exp = gv.gvar(np.mean(expdH_cut), np.std(expdH_cut, ddof = 1)/np.sqrt(len(expdH_cut)))
    #st.markdown(f"""{expdH_exp_bstrap}""")
    fig, ax = plt.subplots()
    ax.hist(expdH_bstrap, bins=25)
    ax.set_title('Bootstrap Distribution')
    ax.set_xlabel('$\exp(-\mathrm{d}\mathcal{H})$')
    ax.set_ylabel('frequency')
    st.pyplot(fig = fig)

    cuts = []
    values = []
    if len(expdH_cut) > 1500:
        print('LEN: ', len(expdH_cut) - 500)
        for cut in range(0, len(expdH_cut) - 500, 250):
            print(cut)
            expdH_bstrap = scipy.stats.bootstrap(
                (expdH_cut[cut::],), np.mean, confidence_level = 0.68
            ).bootstrap_distribution
            expdH_exp_bstrap = gv.dataset.avg_data(expdH_bstrap, bstrap = True)
            cuts.append(cut)
            values.append(expdH_exp_bstrap)
        print(cuts)
        dhp = plot.Plot(h = 2.25)
        dhp.errorbar(cuts, values)
        dhp.decorate(
            #xlim = [0, len(cfgs) - 1],
            ylim = [.85, 1.15],
            xlabel = '$\mathrm{cut}$',
            ylabel = '$\overline{\exp(-\mathrm{d}\mathcal{H})}$'
        )
        #dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
        st.pyplot(fig = dhp.handle)
            
    #print(expdH_exp_bstrap)


    # action change
    try:
        dhp = plot.Plot(h = 2.25)
        ds = np.array(data['kinetic-action-change']) + np.array(data['gauge-action-change']) + \
            np.array(data['fermion-action-change'])
        dsd = [m for idx, m in enumerate(ds) if not data['cut'][idx]]
        plot_hmc_observable(
            dhp,
            ds*expdH,
            '\mathrm{d}\mathcal{S}\exp (-\mathrm{d}\mathcal{H})',
            bins = int(round(0.0025*len(expdH))), 
            range = (min(dsd), max(dsd))
        )
        dhp.decorate(
            xlim = [0, len(cfgs) - 1],
            ylim = [min(dsd), max(dsd)],
            xlabel = '$\mathrm{configuration}$',
            ylabel = '$\mathrm{d}\mathcal{S}\exp (-\mathrm{d}\mathcal{H})$'
        )
        #dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
        st.pyplot(fig = dhp.handle)
        st.markdown("""
        The change in the action weighted by the Boltzmann factor should be Gaussian
        """)
    except ValueError: pass

    st.markdown("### Kinetic action information")
    
    # kinetic action 
    dhp = plot.Plot(h = 2.25)
    plot_hmc_observable(
        dhp, data['kinetic-action'], 'P^2',
        bins = int(round(0.0025*len(expdH))), 
        #range = (0., 2.)
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        #ylim = [0.0, 2.0],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$P^2$'
    )
    dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
    st.pyplot(fig = dhp.handle)

    # kinetic action change
    try:
        dsk = np.array(data['kinetic-action-change'])
        dhp = plot.Plot(h = 2.25)
        plot_hmc_observable(
            dhp, dsk*expdH, '\mathrm{d}P^2',
            bins = int(round(0.0025*len(expdH))), 
            #range = (-2., 2.)
        )
        dhp.decorate(
            xlim = [0, len(cfgs) - 1],
            #ylim = [-2.0, 2.0],
            xlabel = '$\mathrm{configuration}$',
            ylabel = '$\mathrm{d}P^2$'
        )
        dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
        st.pyplot(fig = dhp.handle)
    except ValueError: pass
        
    st.markdown("### Gauge action information")
    
    # gauge action
    dhp = plot.Plot(h = 2.25)
    plot_hmc_observable(
        dhp, data['gauge-action'], '\mathcal{S}_{\mathrm{g}}',
        bins = int(round(0.0025*len(expdH))), 
        #range = (0., 2.)
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        #ylim = [0.0, 2.0],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\mathcal{S}_{\mathrm{g}}$'
    )
    #dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
    st.pyplot(fig = dhp.handle)

    # gauge action change
    try:
        dhp = plot.Plot(h = 2.25)
        dsg = [m for idx, m in enumerate(data['gauge-action-change']) if not data['cut'][idx]]
        plot_hmc_observable(
            dhp,
            np.array(data['gauge-action-change'])*expdH,
            '\mathrm{d}\mathcal{S}_{\mathrm{g}}\exp (-\mathrm{d}\mathcal{H})',
            bins = int(round(0.0025*len(expdH))), 
            range = (min(dsg), max(dsg))
        )
        dhp.decorate(
            xlim = [0, len(cfgs) - 1],
            ylim = [min(dsg), max(dsg)],
            xlabel = '$\mathrm{configuration}$',
            ylabel = '$\mathrm{d}\mathcal{S}_{\mathrm{g}}\exp (-\mathrm{d}\mathcal{H})$'
        )
        #dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
        st.pyplot(fig = dhp.handle)
    except ValueError: pass
        
    st.markdown("### Fermion action information")
    
    # fermion action
    dhp = plot.Plot(h = 2.25)
    sf = np.array(data['fermion-action'])
    plot_hmc_observable(
        dhp, sf, '\mathcal{S}_{\mathrm{f}}',
        bins = int(round(0.0025*len(expdH))), 
        #range = (0., 2.)
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        #ylim = [0.0, 2.0],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\mathcal{S}_{\mathrm{f}}$'
    )
    #dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
    st.pyplot(fig = dhp.handle)

    # fermion action change
    try:
        dhp = plot.Plot(h = 2.25)
        dsf = [m for idx, m in enumerate(data['fermion-action-change']) if not data['cut'][idx]]
        plot_hmc_observable(
            dhp,
            np.array(data['fermion-action-change'])*expdH,
            '\mathrm{d}\mathcal{S}_{\mathrm{f}}\exp (-\mathrm{d}\mathcal{H})',
            bins = int(round(0.0025*len(expdH))), 
            range = (min(dsf), max(dsf))
        )
        dhp.decorate(
            xlim = [0, len(cfgs) - 1],
            ylim = [min(dsf), max(dsf)],
            xlabel = '$\mathrm{configuration}$',
            ylabel = '$\mathrm{d}\mathcal{S}_{\mathrm{f}}\exp (-\mathrm{d}\mathcal{H})$'
        )
        #dhp.axis.axhline(1.0, color = 'k', alpha = 0.25)
        st.pyplot(fig = dhp.handle)
    except ValueError: pass
        
    # average CG per trajectory (fermion)
    st.markdown("""
    ### Long-range thermalization and autocorrelation
    """)
    hcg_lbl = 'avg \\ CG \\ itns \\ per \\ traj'
    dhp = plot.Plot(h = 3.375)
    plot_hmc_observable(
        dhp, fcg, '\\mathrm{' + hcg_lbl + '}',
        bins = int(round(0.005*len(fcg))),
        range = (0., max(fcg))
    )
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        ylim = [0., 2.0*np.mean(fcg)],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\\mathrm{' + hcg_lbl + ' \\ (fermion)}$'
    )
    st.pyplot(fig = dhp.handle)
    st.markdown("""
    The average number of conjugate gradient iterations per trajectory is
    correlated with the chiral condensate, as it is directly related to the
    condition number of the Dirac operator. As a fermionic HMC observable, 
    it is useful for measuring long-distance thermalization and autocorrelations.
    Coloring is same as previous figures.
    """)

    # information
    st.markdown("""
    ## Observables
    Errors and autocorrelation times calculated from 
    implementation of the $\Gamma$-method in [pyerrors](https://github.com/fjosw/pyerrors). 
    - pyerrors: https://doi.org/10.1016/j.cpc.2023.108750
    - $\Gamma$-method: https://doi.org/10.1016/S0010-4655(03)00467-3
    """)

    def plot_observable(h, meas, lbl):
        cfg_cut = [cfg for idx, cfg in enumerate(cfgs) if not data['cut'][idx]] 
        meas_cut = [m for idx, m in enumerate(meas) if not data['cut'][idx]]
        h.scatter(
            [cfg for idx, cfg in enumerate(cfgs) if data['cut'][idx]], 
            [m for idx, m in enumerate(meas) if data['cut'][idx]], 
            **{'facecolor': 'none', 'edgecolor': 'grey', 'alpha': 0.1}
        )
        h.axis.axvline(min(cfg_cut), color = 'k')
        h.scatter(
            cfg_cut, meas_cut, 
            **{'facecolor': 'none', 'edgecolor': 'grey', 'alpha': 0.5}
        )
        (tau, exp) = proper(meas)
        x = [min(cfg_cut), max(cfg_cut) - 1]
        y = [exp, exp]
        h.fill_between(x, y, color = 'magenta', alpha = 0.1)
        h.axis.axhline(exp.mean, color = 'magenta', alpha = 0.5)
        ttl = '$\\overline{' + lbl + '} =' + str(exp) + '$, '
        ttl += '$\\tau = ' + str(tau) + '$'
        h.set_title(ttl)

    plaq = 0.5*(splaq + tplaq)

    srpoly = np.array(data['Re[spatial Polyakov loop]'])
    trpoly = np.array(data['Re[temporal Polyakov loop]'])

    sipoly = np.array(data['Im[spatial Polyakov loop]'])
    tipoly = np.array(data['Im[temporal Polyakov loop]'])

    rpoly = 0.5*(srpoly + trpoly)
    ipoly = 0.5*(sipoly + tipoly)

    poly = np.sqrt(rpoly*rpoly + ipoly*ipoly)

    # plaquette
    dhp = plot.Plot()
    plot_observable(dhp, plaq, '\mathrm{plaquette}')
    dhp.decorate(
        xlim = [0, len(cfgs) - 1],
        ylim = [0.5, 0.9],
        xlabel = '$\mathrm{configuration}$',
        ylabel = '$\mathrm{plaquette}$'
    )
    st.pyplot(fig = dhp.handle)
elif volume == 'all':
    # available couplings
    coupling = st.sidebar.selectbox(
        r"$\beta_{b} \equiv 6/g_{0}^2$", 
        options = BETAS,
        format_func = lambda beta: BETAS[beta]
    )

    # action type toggles per (volume, mass) combination
    st.sidebar.markdown("### Action types")
    action_types = {}
    for _vol in [*VOLUMES][:-1:]:
        if coupling in DATASETS[_vol]:
            for _m in DATASETS[_vol][coupling]:
                _lbl = f"{ENSVOL[_vol]} / {MASSES[_m]}"
                action_types[(_vol, _m)] = st.sidebar.radio(
                    _lbl,
                    ["HISQ", "HYP"],
                    horizontal = True,
                    key = f"action_{_vol}_{coupling}_{_m}"
                )

    edata = {
        'plaquette': {},
        'dH': {},
        'dH2': {},
        'expdH': {},
        'acc-rate': {}
    }

    idxs = []
    
    # get data
    for idx, volume in enumerate([*VOLUMES][:-1:]):
        edata['plaquette'][volume] = {}
        edata['dH'][volume] = {}
        edata['dH2'][volume] = {}
        edata['expdH'][volume] = {}
        edata['acc-rate'][volume] = {}
        idxs.append(idx)
        for mass in DATASETS[volume][coupling]:
            _act = action_types.get((volume, mass), 'HISQ')
            ensemble = ''.join(['f4', ENSVOL[volume], 'b', coupling, 'm', mass, f'_{_act}_pppa'])
            fn = ensemble + '-info.json'
            if os.path.exists(DATA_DIR + fn) and volume != 'all':
                with open(DATA_DIR + fn, 'r') as in_file: data = json.load(in_file)
                cuts = data['cut']

                def filter_cut(d):
                    return [dt for i,dt in enumerate(d) if not cuts[i]]
                
                tplaq = proper(filter_cut(data['temporal plaquette']))[-1]
                splaq = proper(filter_cut(data['spatial plaquette']))[-1]
                
                edata['plaquette'][volume][mass] = 0.5*(tplaq + splaq)

                dH = np.array(filter_cut(data['dH']))
                edata['dH'][volume][mass] = proper(dH)[-1]
                edata['dH2'][volume][mass] = proper(dH*dH)[-1]
                edata['expdH'][volume][mass] = proper(np.exp(-dH))[-1]

                allacc = [1 for idx, _ in enumerate(data['acceptance']) if not data['cut'][idx]]
                accs = [m for idx, m in enumerate(data['acceptance']) if not data['cut'][idx]]
                nrm = np.cumsum(allacc)
                accr = np.cumsum(accs)
                edata['acc-rate'][volume][mass] = accr[-1]/nrm[-1]

    mass_colors = {
        "000"  : "grey",
        "0005" : "magenta",
        "00025": "maroon",
        "0001" : "darkcyan"
    }

    # acceptance rate
    dhp = plot.Plot(h = 2.25)
    for idx, volume in enumerate([*VOLUMES][:-1:]):
        vv = float(volume)
        for mass in DATASETS[volume][coupling]:
            mv = float(MASSES[mass])
            if mass in [*edata['acc-rate'][volume].keys()]:
                dhp.scatter([vv], [edata['acc-rate'][volume][mass]], **{
                    'color': mass_colors[mass]
                })
    dhp.decorate(
        #xlim = [0, len(cfgs) - 1],
        ylim = [0.0, 1.0],
        xlabel = '$L/a$',
        ylabel = '$\mathrm{acceptance \ rate}$'
    )
    st.pyplot(fig = dhp.handle)

    # dH
    dhp = plot.Plot(h = 2.25)
    for idx, volume in enumerate([*VOLUMES][:-1:]):
        vv = float(volume)
        for mass in DATASETS[volume][coupling]:
            mv = float(MASSES[mass])
            if mass in [*edata['dH'][volume].keys()]:
                dhp.errorbar([vv], [edata['dH'][volume][mass]], **{
                    'color': mass_colors[mass],
                    'fmt': 'o',
                    'capsize': 4.,
                    
                })
    dhp.decorate(
        #xlim = [0, len(cfgs) - 1],
        ylim = [-1.0, 1.0],
        xlabel = '$L/a$',
        ylabel = '$\mathrm{d}\mathcal{H}$'
    )
    st.pyplot(fig = dhp.handle)

    # dH2
    dhp = plot.Plot(h = 2.25)
    for idx, volume in enumerate([*VOLUMES][:-1:]):
        vv = float(volume)
        for mass in DATASETS[volume][coupling]:
            mv = float(MASSES[mass])
            if mass in [*edata['dH2'][volume].keys()]:
                dhp.errorbar([vv], [edata['dH2'][volume][mass]], **{
                    'color': mass_colors[mass],
                    'fmt': 'o',
                    'capsize': 4.,
                    
                })
    dhp.decorate(
        #xlim = [0, len(cfgs) - 1],
        ylim = [-0.1, 2.0],
        xlabel = '$L/a$',
        ylabel = '$\mathrm{d}\mathcal{H}^2$'
    )
    st.pyplot(fig = dhp.handle)

    # expdH
    dhp = plot.Plot(h = 2.25)
    for idx, volume in enumerate([*VOLUMES][:-1:]):
        vv = float(volume)
        for mass in DATASETS[volume][coupling]:
            mv = float(MASSES[mass])
            if mass in [*edata['expdH'][volume].keys()]:
                dhp.errorbar([vv], [edata['expdH'][volume][mass]], **{
                    'color': mass_colors[mass],
                    'fmt': 'o',
                    'capsize': 4.,
                    
                })
    dhp.decorate(
        #xlim = [0, len(cfgs) - 1],
        ylim = [0.9, 1.1],
        xlabel = '$L/a$',
        ylabel = '$\exp(-\mathrm{d}\mathcal{H})$'
    )
    st.pyplot(fig = dhp.handle)

    # plaquette
    dhp = plot.Plot(h = 2.25)
    for idx, volume in enumerate([*VOLUMES][:-1:]):
        vv = float(volume)
        for mass in DATASETS[volume][coupling]:
            mv = float(MASSES[mass])
            if mass in [*edata['plaquette'][volume].keys()]:
                dhp.errorbar([vv], [edata['plaquette'][volume][mass]], **{
                    'color': mass_colors[mass],
                    'fmt': 'o',
                    'capsize': 4.,
                    
                })
    dhp.decorate(
        #xlim = [0, len(cfgs) - 1],
        #ylim = [0.5, 1.0],
        xlabel = '$L/a$',
        ylabel = '$\mathrm{plaquette}$'
    )
    st.pyplot(fig = dhp.handle)


            
else: st.image("ensemble_not_found.png")   
