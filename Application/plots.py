import numpy as np

def plot_variables_over_grid(ax, variables, grid, x_label=None, **kwargs):

    """
    Plots specified variables over a given grid on the provided axes.
    
    Parameters:
    - ax: list of matplotlib axes to plot on
    - model: model object containing solution data
    - variables: dict mapping variable names to their data arrays
    - grid: array of x values for the plot
    - x_label: label for the x-axis (optional)
    - kwargs: additional keyword arguments for plotting
    """
    
    # check if ax has enough axes
    if len(ax) < len(variables):
        raise ValueError(f"ax must be a list of {len(variables)} or more axes")
    
    # plot each variable
    for i, (var_name, var_data) in enumerate(variables.items()):
        ax[i].plot(grid, var_data, **kwargs)
        ax[i].set_title(var_name)
        if x_label:
            ax[i].set_xlabel(x_label)
        if i == 0:
            ax[i].legend()
    
    return ax


def plot_choices_over_time_female_single_to_single(ax, model, iA, **kwargs):
    
    # settings
    if 'label' not in kwargs:
        kwargs['label'] = f'female s->s, A={model.par.grid_Aw[iA]:.2f}'
    grid = np.arange(0,model.par.T)
    x_label = '$t$'
    variables = {
        'C_tot':    model.sol.Cw_tot_single_to_single[:,iA],
        'l':        [0] * len(grid), #model.sol.l[:,iA],
        'C_priv':   model.sol.Cw_priv_single_to_single[:,iA],
        'h':        model.sol.hw_single_to_single[:,iA],
        'C_inter':  model.sol.Cw_inter_single_to_single[:,iA],
        'Q':        model.sol.Qw_single_to_single[:,iA]
    }
    
    # plot
    plot_variables_over_grid(ax, variables, grid, x_label=x_label, **kwargs)
    
    return ax


def plot_choices_over_time_male_single_to_single(ax, model, iA, **kwargs):
    
    # settings
    if 'label' not in kwargs:
        kwargs['label'] = f'male s->s, A={model.par.grid_Aw[iA]:.2f}'
    grid = np.arange(0,model.par.T)
    x_label = '$t$'
    variables = {
        'C_tot':    model.sol.Cm_tot_single_to_single[:,iA],
        'l':        [0] * len(grid), #model.sol.l[:,iA],
        'C_priv':   model.sol.Cm_priv_single_to_single[:,iA],
        'h':        model.sol.hm_single_to_single[:,iA],
        'C_inter':  model.sol.Cm_inter_single_to_single[:,iA],
        'Q':        model.sol.Qm_single_to_single[:,iA]
    }
    
    # plot
    plot_variables_over_grid(ax, variables, grid, x_label=x_label, **kwargs)

    return ax


def plot_choices_over_assets_female_single_to_single(ax, model, t, **kwargs):
    
    # settings
    if 'label' not in kwargs:
        kwargs['label'] = f'female s->s, t={t}'    
    grid = model.par.grid_Aw
    x_label = '$Assets$'
    variables = {
        'C_tot':    model.sol.Cw_tot_single_to_single[t, :],
        'l':        [0] * len(grid), #model.sol.l[t, :],
        'C_priv':   model.sol.Cw_priv_single_to_single[t, :],
        'h':        model.sol.hw_single_to_single[t, :],
        'C_inter':  model.sol.Cw_inter_single_to_single[t, :],
        'Q':        model.sol.Qw_single_to_single[t, :]
    }
    
    # plot
    plot_variables_over_grid(ax, variables, grid, x_label=x_label, **kwargs)

    return ax


def plot_choices_over_assets_male_single_to_single(ax, model, t, **kwargs):
    
    # settings
    if 'label' not in kwargs:
        kwargs['label'] = f'male s->s, t={t}'    
    grid = model.par.grid_Am
    x_label = 'Assets'
    variables = {
        'C_tot':    model.sol.Cm_tot_single_to_single[t, :],
        'l':        [0] * len(grid), #model.sol.l[t, :],
        'C_priv':   model.sol.Cm_priv_single_to_single[t, :],
        'h':        model.sol.hm_single_to_single[t, :],
        'C_inter':  model.sol.Cm_inter_single_to_single[t, :],
        'Q':        model.sol.Qm_single_to_single[t, :]
    }
    
    # plot
    plot_variables_over_grid(ax, variables, grid, x_label=x_label, **kwargs)

    return ax