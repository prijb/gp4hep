# Uses a mean function to motivate the GPR (note: Still doesn't work well for large amounts of data)
import ROOT
import scipy.linalg
ROOT.EnableImplicitMT()
import os, sys, glob, pickle, argparse, subprocess, multiprocessing, itertools, yaml
import numpy as np
import numba
import scipy
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from iminuit import Minuit
from iminuit import minimize as MinuitMinimize
from iminuit.cost import LeastSquares 
from scipy.optimize import minimize as ScipyMinimize
from functools import partial
from scipy.stats import chi2, norm
import random
# plt.style.use(hep.style.ROOT)
plt.style.use(hep.style.CMS)
#plt.rcParams["figure.figsize"] = (12.5, 10)
#plt.rcParams["figure.figsize"] = (12.5, 15.0)
cols = ['#4285f4','#ea4335','#fbbc05','#34a853', '#a00498', '#536267']

# Repo imports
from mean_functions import *
from kernels import *

######### Stats ###############
def get_chi2(y, y_err, y_pred, y_pred_err, use_pred_err=False):
    """
    Calculate the chi2 value for the given data and model predictions.
    """
    # Filter out zeroes from the predictions
    zero_mask = (y == 0)
    y = y[~zero_mask]
    y_err = y_err[~zero_mask]
    y_pred = y_pred[~zero_mask]
    y_pred_err = y_pred_err[~zero_mask]
    # Calculate the chi2 value
    if use_pred_err:
        chi2 = np.sum(((y - y_pred) / np.sqrt(y_pred_err**2 + y_err**2))**2)
    else:
        chi2 = np.sum(((y - y_pred) / y_err)**2)
    return chi2

############# GPR likelihood definition ################
def GP_noise(params,X1, y1, X2, kernel_func, noise):
    """
    Calculate the posterior mean (non-parametric part) and covariance matrix for y2
    based on the corresponding input X2, the noisy observations 
    (y1, X1), and the prior kernel function.
    """
    if config["kernel_settings"]["exponentiate_params"]:  params = [np.exp(param) for param in params]
    K11 = kernel_func(X1, X1,*params) + np.diag(noise**2)
    K12 = kernel_func(X1, X2,*params)
    # Does K21 * (K11^-1) (solve does K11^-1 * K12)
    solved = scipy.linalg.solve(K11, K12, assume_a='pos').T    
    r = y1
    μ2 = solved @ r

    K22 = kernel_func(X2, X2,*params)
    K2 = K22 - (solved @ K12)
    return μ2, np.sqrt(np.diag(K2))

def gp_lml(params, x, y, noise, bkg_func, n_bkg_params, sig_func, n_sig_params, kernel_func):
    X = x[:, None]
    Y = y[:, None]
    y_prior_bkg = np.zeros_like(Y)
    y_prior_sig = np.zeros_like(Y)

    bkg_params = None
    signal_params = None

    if n_bkg_params > 0:
        bkg_params = params[:n_bkg_params]
        y_prior_bkg = bkg_func(X, *bkg_params)

    if n_sig_params > 0:
        signal_params = params[n_bkg_params:n_bkg_params+n_sig_params]
        y_prior_sig = sig_func(X.reshape(-1), *signal_params)[:, None]

    kernel_params = params[n_bkg_params+n_sig_params:-1]
    
    # Exponentiate params
    if config["kernel_settings"]["exponentiate_params"]: kernel_params = [np.exp(kernel_param) for kernel_param in kernel_params]

    YFIT = Y - y_prior_bkg - params[-1]*y_prior_sig
    K11 = kernel_func(X, X, *kernel_params) + np.diag(noise**2)
    _, K11_logdet = np.linalg.slogdet(K11)
    return 2.0*(0.5*(YFIT.T @ np.linalg.inv(K11) @ YFIT) + 0.5*K11_logdet + 0.5*np.float64(Y.shape[0])*np.log(2*np.pi))

# Using a signal template
def gp_template_lml(params, x, y, noise, bkg_func, n_bkg_params, ys, kernel_func):
    X = x[:, None]
    Y = y[:, None]
    YS = ys[:, None]
    y_prior_bkg = np.zeros_like(Y)

    bkg_params = None

    if n_bkg_params > 0:
        bkg_params = params[:n_bkg_params]
        y_prior_bkg = bkg_func(X, *bkg_params)

    kernel_params = params[n_bkg_params:-1]
    
    # Exponentiate params
    if config["kernel_settings"]["exponentiate_params"]: kernel_params = [np.exp(kernel_param) for kernel_param in kernel_params]
    
    YFIT = Y - y_prior_bkg - params[-1]*YS
    K11 = kernel_func(X, X, *kernel_params) + np.diag(noise**2)
    _, K11_logdet = np.linalg.slogdet(K11)
    return 2.0*(0.5*(YFIT.T @ np.linalg.inv(K11) @ YFIT) + 0.5*K11_logdet + 0.5*np.float64(Y.shape[0])*np.log(2*np.pi))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='resolved2016_reg2.root')
    parser.add_argument("--input_file_sig",default='signal.root')
    parser.add_argument('--output', default='test')
    parser.add_argument('--config', default="fit_config.yaml")
    args = parser.parse_args()

    plot_dir = f"plots/{args.output}"
    os.makedirs(plot_dir, exist_ok=True)

    config = None
    with open(args.config, 'r') as config_stream:
        config = yaml.full_load(config_stream)

    signal_mass = config["general"]["signal_mass"]
    signal_yield = 421817
    if signal_mass == 250: signal_yield = 421817
    elif signal_mass == 350: signal_yield = 186432
    elif signal_mass == 450: signal_yield = 417989
    elif signal_mass == 500: signal_yield = 469523
    elif signal_mass == 600: signal_yield = 379415
    else: 
        print("Error: Invalid signal mass")
        sys.exit(1)
    signal_lumiscale = config["general"]["lumi"] * config["general"]["xs"] / signal_yield

    ######## Data loading ######
    f = ROOT.TFile(args.input_file)
    h = f.Get(config["general"]["hist_name_data"])
    h.Rebin(config["general"]["rebin"])
    
    # Data
    x_edges=[]
    x=[]
    y=[]
    dy=[]

    for i in range(1,h.GetNbinsX()+1):
        if(h.GetBinContent(i)>-1):
            x_edges.append(h.GetBinLowEdge(i))
            x.append(h.GetBinCenter(i))
            y.append(h.GetBinContent(i))
            dy.append(h.GetBinErrorUp(i))
    x_edges.append(h.GetBinLowEdge(h.GetNbinsX()+1))

    x = np.array(x)
    y = np.array(y)
    dy = np.array(dy)
    x_edges = np.array(x_edges)
    dx = x_edges[1] - x_edges[0]

    # Signal
    x_edges_sig=[]
    x_sig=[]
    y_sig=[]
    dy_sig=[]

    f_sig = ROOT.TFile(args.input_file_sig)
    h_sig = f_sig.Get(config["general"]["hist_name_signal"])
    h_sig.Rebin(config["general"]["rebin"])
    h_sig.Scale(signal_lumiscale)

    for i in range(1,h_sig.GetNbinsX()+1):
        if(h_sig.GetBinContent(i)>-1):
            x_edges_sig.append(h_sig.GetBinLowEdge(i))
            x_sig.append(h_sig.GetBinCenter(i))
            y_sig.append(h_sig.GetBinContent(i))
            dy_sig.append(h_sig.GetBinErrorUp(i))
    
    x_edges_sig.append(h_sig.GetBinLowEdge(h_sig.GetNbinsX()+1))
    x_edges_sig = np.array(x_edges_sig)
    x_sig = np.array(x_sig)
    y_sig = np.array(y_sig)
    dy_sig = np.array(dy_sig)

    # Scale data to pb/GeV if needed
    if config["fit_settings"]["scale_data"]:
        print(f"Scaling data to pb/GeV units")
        x_widths = x_edges[1:] - x_edges[:-1]
        y = y/(x_widths * config["general"]["lumi"])
        y_sig = y_sig/(x_widths * config["general"]["lumi"])
        dy = dy/(x_widths * config["general"]["lumi"])
        dy_sig = dy_sig/(x_widths * config["general"]["lumi"])

    # Inject signal if asked for   
    if (config["general"]["inject_signal"] != 0):
        y = y + (config["general"]["inject_signal"]*y_sig)

    # Filter by range
    fit_range = config["fit_settings"]["range"]
    fit_range_down = fit_range[0]
    fit_range_up = fit_range[1]

    if fit_range_down is None: fit_range_down = x_edges[0]
    if fit_range_up is None: fit_range_up = x_edges[0]

    print(f"\nRestricting data to range [{fit_range_down}, {fit_range_up}]")
    mask = (x > fit_range_down) & (x < fit_range_up)

    x = x[mask]
    y = y[mask]
    y_sig = y_sig[mask]
    dy = dy[mask]
    dy_sig = dy_sig[mask]
    x_edges = np.append((x - 0.5*dx), [x[-1] + 0.5*dx])

    ####### Background only fit ########
    bkg_func = None
    bkg_func_name = config["fit_settings"]["background_function"]  
    if bkg_func_name == "mod_exp":
        print("\nUsing modified exponential for bkg")
        bkg_func = mod_exp_simpson
    elif bkg_func_name == "poly_ext":
        print("\nUsing extended polynomial for bkg")
        bkg_func = poly_ext_simpson
    elif bkg_func_name == None:
        print("\nNo mean function passed")
    else:
        print(f"\nInvalid function {bkg_func_name}")
        sys.exit(2)

    params_bkg = dict()
    bounds_bkg = dict()

    if bkg_func_name is not None:
        params_bkg = config["fit_settings"][bkg_func_name]["init"]
        bounds_bkg = config["fit_settings"][bkg_func_name]["bounds"]

        #### Manually set some bounds 
        #if bkg_func_name == "mod_exp":
        #    params_bkg["p0"] = np.sum(y)
        
        # Use this to guess the value of p0 
        print(f"Starting with p0 of {params_bkg["p0"]}")
        params_bkg["p0"] = 1.0
        norm_factor_start = np.trapz(bkg_func(x, **params_bkg), x=x)
        print(f"p0 is {norm_factor_start:.3e} for unit area, multiplying by integral {np.sum(y)*dx:.3e} to get p0 guess of {norm_factor_start*np.sum(y)*dx:.3e}")
        params_bkg["p0"] = norm_factor_start*np.sum(y)*dx

        # Fit
        print("\nMinimizing bkg function")
        loss_bkg = LeastSquares(x, y, dy, bkg_func)
        m_bkg = Minuit(loss_bkg, **params_bkg)
        for param_bkg in params_bkg.keys():
            m_bkg.limits[param_bkg] = bounds_bkg[param_bkg]
        m_bkg.migrad()
        m_bkg.hesse()
        print(m_bkg)

        for param_bkg in params_bkg.keys():
            params_bkg[param_bkg] = m_bkg.values[param_bkg]
        y_fit_bkg = bkg_func(x,**params_bkg)

        # Plot
        plt.rcParams["figure.figsize"] = (12.5, 10.0)
        fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.1), sharex=True)
        hep.histplot(y, bins=x_edges, yerr=dy, ax=axs[0], label=f"Data", histtype='errorbar', color='black', density=False)
        axs[0].plot(x, y_fit_bkg, label=f"Functional postfit", color="blue")
        axs[0].set_xlabel("")
        axs[0].set_ylabel(f"Events/{dx} GeV")
        axs[0].set_xlim(x_edges[0], x_edges[-1])
        axs[0].set_yscale('log')
        axs[0].legend()
        # Plot the residuals
        hep.histplot(y - y_fit_bkg, bins=x_edges, yerr=dy, ax=axs[1], label=f"Data", histtype='errorbar', color='black', density=False)
        axs[1].set_xlabel("Mass [GeV]")
        axs[1].set_ylabel("Data - Param")
        axs[1].axhline(0, color='black', linestyle='--')
        hep.cms.label(data=True, llabel="Private Work", rlabel=r"Level-1 Scouting", ax=axs[0])
        plt.savefig(f"{plot_dir}/bkg_only_fit.png")

    ####### Signal only fit ########
    sig_func = None
    sig_func_name = config["fit_settings"]["signal_function"]  
    if sig_func_name == "DSCB":
        print("\nUsing DSCB for sig")
        sig_func = DSCB_pdf
    else:
        print(f"\nInvalid function {sig_func_name}")
        sys.exit(2)

    params_sig = config["fit_settings"][sig_func_name]["init"]
    bounds_sig = config["fit_settings"][sig_func_name]["bounds"]

    # Manually set some bounds
    params_sig["a"] = np.sum(y_sig)
    bounds_sig["mu"] = [max(signal_mass - 100, x_edges[0]), min(signal_mass + 100, x_edges[-1])]

    # Fit
    print("\nMinimizing sig function")
    loss_sig = LeastSquares(x, y_sig, dy_sig, sig_func)
    m_sig = Minuit(loss_sig, **params_sig)
    for param_sig in params_sig.keys():
        m_sig.limits[param_sig] = bounds_sig[param_sig]
    m_sig.migrad()
    m_sig.hesse()
    print(m_sig)

    for param_sig in params_sig.keys():
        params_sig[param_sig] = m_sig.values[param_sig]
        ## Fix the signal params
        bounds_sig[param_sig] = [m_sig.values[param_sig], m_sig.values[param_sig]]
    y_fit_sig = sig_func(x,**params_sig)

    # Plot
    plt.rcParams["figure.figsize"] = (12.5, 10.0)
    fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.1), sharex=True)
    hep.histplot(y_sig, bins=x_edges, yerr=dy_sig, ax=axs[0], label=f"Signal Prefit (r = 1)", histtype='step', color='red', density=False)
    axs[0].plot(x, y_fit_sig, label=f"Signal postfit", color="red")
    axs[0].set_xlabel("")
    axs[0].set_ylabel(f"Events/{dx} GeV")
    axs[0].set_xlim(x_edges[0], x_edges[-1])
    #axs[0].set_yscale('log')
    axs[0].legend()
    # Plot the residuals
    hep.histplot(y_sig - y_fit_sig, bins=x_edges, yerr=dy_sig, ax=axs[1], label=f"Signal", histtype='errorbar', color='black', density=False)
    axs[1].set_xlabel("Mass [GeV]")
    axs[1].set_ylabel("Signal - Fit")
    axs[1].axhline(0, color='black', linestyle='--')
    hep.cms.label(data=True, llabel="Private Work", rlabel=r"Level-1 Scouting", ax=axs[0])
    plt.savefig(f"{plot_dir}/sig_only_fit.png")

    ####### GPR fit ########
    kernel_func = None
    kernel_func_name = config["kernel_settings"]["kernel"]
    if kernel_func_name == "RBF":
        print("\nUsing RBF kernel")
        kernel_func = exponentiated_quadratic
    elif kernel_func_name == "Gibbs":
        print("\nUsing Gibbs kernel")
        kernel_func = modified_gibbs
    else:
        print(f"\nInvalid kernel {kernel_func_name}")
        sys.exit(3)

    params_kernel = config["kernel_settings"][kernel_func_name]["init"]
    bounds_kernel = config["kernel_settings"][kernel_func_name]["bounds"]

    if config["kernel_settings"]["exponentiate_params"]:
        for param_kernel in params_kernel.keys():
            params_kernel[param_kernel] = np.log(params_kernel[param_kernel])
            
            new_bound_kernel = []
            for bound in bounds_kernel[param_kernel]:
                if bound is None:
                    new_bound_kernel.append(None)
                else:
                    print(float(bound))
                    new_bound_kernel.append(np.log(float(bound)))
            bounds_kernel[param_kernel] = new_bound_kernel

    # GPR params depends on choice of signal treatment
    if bkg_func_name is not None:
        if config["fit_settings"]["signal"] == 'template':
            print("\nUsing signal template in GPR")
            params = [params_bkg[param_name] for param_name in params_bkg.keys()] + [params_kernel[param_name] for param_name in params_kernel.keys()] + [1]
            bounds = [bounds_bkg[bound_name] for bound_name in bounds_bkg.keys()] + [bounds_kernel[bound_name] for bound_name in bounds_kernel.keys()] + [[-20, 20]]
        elif config["fit_settings"]["signal"] == 'fit':
            print("\nUsing signal DSCB fit in GPR")
            params = [params_bkg[param_name] for param_name in params_bkg.keys()] + [params_sig[param_name] for param_name in params_sig.keys()] + [params_kernel[param_name] for param_name in params_kernel.keys()] + [1]
            bounds = [bounds_bkg[bound_name] for bound_name in bounds_bkg.keys()] + [bounds_sig[bound_name] for bound_name in bounds_sig.keys()] + [bounds_kernel[bound_name] for bound_name in bounds_kernel.keys()] + [[-20, 20]]
        else:
            print("\nUsing background only (r=0) fit in GPR")
            params = [params_bkg[param_name] for param_name in params_bkg.keys()] + [params_kernel[param_name] for param_name in params_kernel.keys()] + [0]
            bounds = [bounds_bkg[bound_name] for bound_name in bounds_bkg.keys()] + [bounds_kernel[bound_name] for bound_name in bounds_kernel.keys()] + [[0, 0]]
    else:
        if config["fit_settings"]["signal"] == 'template':
            print("\nUsing signal template in GPR")
            params = [params_kernel[param_name] for param_name in params_kernel.keys()] + [1]
            bounds = [bounds_kernel[bound_name] for bound_name in bounds_kernel.keys()] + [[-20, 20]]
        elif config["fit_settings"]["signal"] == 'fit':
            print("\nUsing signal DSCB fit in GPR")
            params = [params_sig[param_name] for param_name in params_sig.keys()] + [params_kernel[param_name] for param_name in params_kernel.keys()] + [1]
            bounds = [bounds_sig[bound_name] for bound_name in bounds_sig.keys()] + [bounds_kernel[bound_name] for bound_name in bounds_kernel.keys()] + [[-20, 20]]
        else:
            print("\nUsing background only (r=0) fit in GPR")
            params = [params_kernel[param_name] for param_name in params_kernel.keys()] + [0]
            bounds = [bounds_kernel[bound_name] for bound_name in bounds_kernel.keys()] + [[0, 0]]


    print("\nFitting GPR with")
    print(f"Params: {params}")
    print(f"Bounds: {bounds}")

    if config["fit_settings"]["signal"] == 'fit':
        result = MinuitMinimize(
            gp_lml,
            params, 
            args = (x, y, dy, bkg_func, len(params_bkg.keys()), sig_func, len(params_sig.keys()), kernel_func),
            bounds = bounds,
            options = {'disp':False},
            method = "migrad"
        )
    else:
        result = MinuitMinimize(
            gp_template_lml,
            params, 
            args = (x, y, dy, bkg_func, len(params_bkg.keys()), y_sig, kernel_func),
            bounds = bounds,
            options = {'disp':False},
            method = "migrad"
        )
    
    fit_params = result.x
    fit_params_minuit = result.minuit
    fit_params_err = fit_params_minuit.errors
    print("GPR Minuit result: ", result)
    print(f"\nFitted strength: {fit_params[-1]:.2f} +/- {fit_params_err[-1]:.3f}")

    bkg_func_params = []
    if bkg_func_name is not None:
        bkg_func_params = fit_params[:len(params_bkg)]
    #kernel_params = fit_params[len(params_bkg) + len(params_sig):-1]

    if config["fit_settings"]["signal"] == 'template': 
        sig_func_params = []
        kernel_params = fit_params[len(params_bkg):-1]
    elif config["fit_settings"]["signal"] == 'fit': 
        sig_func_params = fit_params[len(params_bkg):len(params_bkg) + len(params_sig)]
        kernel_params = fit_params[len(params_bkg) + len(params_sig):-1]
    else: 
        sig_func_params = []
        kernel_params = fit_params[len(params_bkg):-1]

    y_fit_bkg = np.zeros_like(y)

    if bkg_func_name is not None:
        y_fit_bkg = bkg_func(x, *bkg_func_params)
    if config["fit_settings"]["signal"] == 'template': y_fit_sig = y_sig * fit_params[-1]
    elif config["fit_settings"]["signal"] == 'fit': y_fit_sig = sig_func(x, *sig_func_params) * fit_params[-1]
    else: y_fit_sig = np.zeros_like(y)

    yp, yp_err = GP_noise(kernel_params, x[:, None], y - y_fit_bkg - y_fit_sig, x[:, None], kernel_func, dy)
    yp_bkg = yp + y_fit_bkg
    yp_tot = yp_bkg + y_fit_sig
    
    chi2 = get_chi2(y, dy, yp_tot, yp_err, use_pred_err=True)
    ndf = len(y) - len(fit_params)


    ## Plot GPR performance at fitting the residuals
    y_residual = y - y_fit_bkg
    fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.1), sharex=True)
    hep.histplot(y_residual, bins=x_edges, yerr=dy, ax=axs[0], label=f"Data", histtype='errorbar', color='black', density=False)
    axs[0].plot(x, yp, label="GPR fit (bkg,sig subtracted)", color='blue', zorder=10)
    if not (config["fit_settings"]["signal"] == 'zero'): 
        hep.histplot(y_fit_sig, bins=x_edges, yerr=dy_sig, ax=axs[0], label=f"Signal (r={fit_params[-1]:.2f}+/-{fit_params_err[-1]:.3f})", histtype='step', color='red', density=False)
    axs[0].fill_between(x, yp-yp_err, yp+yp_err, alpha=0.5, color='blue')
    axs[0].text(0.55, 0.85, f"$\chi^{2}/ndf$: {chi2:.2f}/{ndf} = {chi2/ndf:.2f}", fontsize=16, transform=axs[0].transAxes)
    axs[0].set_xlabel("")
    axs[0].set_ylabel(f"Data - Param")
    axs[0].set_xlim(x_edges[0], x_edges[-1])
    axs[0].legend()
    # Pulls
    bar_widths = x_edges[1:] - x_edges[:-1]
    pulls_num = (y_residual - yp - y_fit_sig)
    pulls_den = np.sqrt(yp_err**2 + dy**2)
    pulls_prefit = y_residual/dy
    pulls = pulls_num / pulls_den
    axs[1].bar(x, pulls_prefit, width=bar_widths, color='red', alpha=0.5, label='Pulls (bkgfunc only)')
    axs[1].bar(x, pulls, width=bar_widths, color='blue', alpha=0.5, label='Pulls')
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_xlabel("Mjj")
    axs[1].set_ylabel("Pulls")
    axs[1].set_ylim(-5, 5)
    axs[1].legend(fontsize=16)
    hep.cms.label(data=True, llabel="Private Work", rlabel=r"Level-1 Scouting", ax=axs[0])
    plt.savefig(f"{plot_dir}/postfit_residual_pulls.png")

    # Plot postfit (pulls)
    fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.1), sharex=True)
    hep.histplot(y, bins=x_edges, yerr=dy, ax=axs[0], label=f"Data", histtype='errorbar', color='black', density=False)
    axs[0].plot(x, yp_bkg, label="Background prediction", color='blue', zorder=10)
    if not (config["fit_settings"]["signal"] == 'zero'): 
        if fit_params[-1] > 0: 
            hep.histplot(y_fit_sig, bins=x_edges, yerr=dy_sig, ax=axs[0], label=f"Signal (r={fit_params[-1]:.2f}+/-{fit_params_err[-1]:.3f})", histtype='step', color='red', density=False)    
    axs[0].fill_between(x, yp_bkg-yp_err, yp_bkg+yp_err, alpha=0.5, color='blue')
    axs[0].text(0.55, 0.85, f"$\chi^{2}/ndf$: {chi2:.2f}/{ndf} = {chi2/ndf:.2f}", fontsize=16, transform=axs[0].transAxes)
    axs[0].set_xlabel("")
    axs[0].set_ylabel(f"Events/{dx} GeV")
    axs[0].set_xlim(x_edges[0], x_edges[-1])
    axs[0].set_yscale('log')
    axs[0].legend()
    # Pulls
    bar_widths = x_edges[1:] - x_edges[:-1]
    pulls_num = (y - yp_tot)
    pulls_den = np.sqrt(yp_err**2 + dy**2)
    pulls = pulls_num / pulls_den
    # Pulls with only param function
    pulls_param = (y - y_fit_bkg)/dy
    axs[1].bar(x, pulls_param, width=bar_widths, color='red', alpha=0.5, label='Pulls (bkg func only)')
    axs[1].bar(x, pulls, width=bar_widths, color='blue', alpha=0.5, label='Pulls')
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_xlabel("Mjj")
    axs[1].set_ylabel("Pulls")
    axs[1].set_ylim(-5, 5)
    axs[1].legend(fontsize=16)
    hep.cms.label(data=True, llabel="Private Work", rlabel=r"Level-1 Scouting", ax=axs[0])
    plt.savefig(f"{plot_dir}/postfit_pulls.png")

    # Plot postfit (Residuals with signal overlaid)
    if not (config["fit_settings"]["signal"] == 'zero'): 
        max_mag_sig = np.max(np.abs(y_fit_sig))
        fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
        hep.histplot(y, bins=x_edges, yerr=dy, ax=axs[0], label=f"Data", histtype='errorbar', color='black', density=False)
        axs[0].plot(x, yp_bkg, label="Background prediction", color='blue', zorder=10)
        if fit_params[-1] > 0: 
            hep.histplot(y_fit_sig, bins=x_edges, yerr=dy_sig, ax=axs[0], label=f"Signal (r={fit_params[-1]:.2f}+/-{fit_params_err[-1]:.3f})", histtype='step', color='red', density=False)    
        axs[0].fill_between(x, yp_bkg-yp_err, yp_bkg+yp_err, alpha=0.5, color='blue')
        axs[0].text(0.55, 0.85, f"$\chi^{2}/ndf$: {chi2:.2f}/{ndf} = {chi2/ndf:.2f}", fontsize=16, transform=axs[0].transAxes)
        axs[0].set_xlabel("")
        axs[0].set_ylabel(f"Events/{dx} GeV")
        axs[0].set_xlim(x_edges[0], x_edges[-1])
        axs[0].set_yscale('log')
        axs[0].legend()
        # Pulls
        bar_widths = x_edges[1:] - x_edges[:-1]
        resid = y - yp_bkg
        axs[1].axhline(0, color='black', linestyle='--', lw=1)
        hep.histplot(resid, bins=x_edges, yerr=np.sqrt(dy**2 + yp_err**2), ax=axs[1], histtype='errorbar', color='black', density=False)
        hep.histplot(y_fit_sig, bins=x_edges, ax=axs[1], label=f"Signal (r={fit_params[-1]:.2f}+/-{fit_params_err[-1]:.3f})", histtype='step', color='red', density=False)
        axs[1].set_xlabel("Mjj")
        axs[1].set_ylabel("Residuals")
        axs[1].set_ylim(-3*max_mag_sig, 3*max_mag_sig)
        hep.cms.label(data=True, llabel="Private Work", rlabel=r"Level-1 Scouting", ax=axs[0])
        plt.savefig(f"{plot_dir}/postfit_residual.png")

    ####### Parametric S+B fit for comparison ########
    if config["fit_settings"]["signal"] == 'fit':
        print("Fitting S + B for parametric model")
        def parametric_model(x, p0, p1, p2, p3, p4, a, mu, sigma, n_low, alpha_low, n_high, alpha_high, r):
            b = bkg_func(x, p0, p1, p2, p3, p4)
            s = r * sig_func(x, a, mu, sigma, n_low, alpha_low, n_high, alpha_high)
            return (s + b)

        #params_parametric = [params_bkg[param_name] for param_name in params_bkg.keys()] + [params_sig[param_name] for param_name in params_sig.keys()] + [1]
        #bounds_parametric = [bounds_bkg[bound_name] for bound_name in bounds_bkg.keys()] + [bounds_sig[bound_name] for bound_name in bounds_sig.keys()] + [[-20, 20]]

        params_parametric = dict(
            r = 1,
            **params_bkg,
            **params_sig,
        )

        bounds_parametric = dict(
            r = [-20, 20],
            **bounds_bkg,
            **bounds_sig,
        )

        loss_parametric_model = LeastSquares(x, y, dy, parametric_model)
        m_sb = Minuit(loss_parametric_model, **params_parametric)
        for param_parametric in params_parametric.keys():
            m_sb.limits[param_parametric] = bounds_parametric[param_parametric]
        m_sb.migrad()
        m_sb.hesse()
        print(m_sb)

        print(f"\nFitted strength (parametric): {m_sb.values["r"]:.2f} +/- {m_sb.errors["r"]:.3f}")