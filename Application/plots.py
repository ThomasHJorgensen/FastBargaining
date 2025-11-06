import numpy as np
import pandas as pd

class model_plotter():
    def __init__(self, model, model_name=None, titles=['variable'], labels=['variable', 'index',], grid_names=None, **kwargs):
        
        self.model = model
        self.model_name = model_name if model_name is not None else model.name
        self.index_labels_per_row = kwargs.get('index_labels_per_row', 3)  # Default to 3 if not provided
        self.sim_data = self.sim_to_df()
        
        if isinstance(titles, str):
            self.titles = [titles]
        elif isinstance(titles, list):
            self.titles = titles
        else:
            raise ValueError("Titles must be a string or a list of strings")
        
        if isinstance(labels, str):
            self.labels = [labels]
        elif isinstance(labels, list):
            self.labels = labels
        else:
            raise ValueError("Labels must be a string or a list of strings")

        if grid_names is None:
            # Default grid names if not provided
            self.grid_names = {
                't': 'grid_t', 
                'iK': 'grid_K',
                'iP': 'grid_power', 
                'iL': 'grid_love', 
                'iA': 'grid_A',
                'iA_pd': 'grid_A_pd',
                'il': 'grid_l',
            }
        elif isinstance(grid_names, dict):
            self.grid_names = grid_names
        else:
            raise ValueError("grid_names must be a dictionary with index IDs as keys and grid names as values")
        
        self.grids = self.make_grid_dict(self.grid_names)
        self.grid_lengths = self.make_grid_lengths_dict(self.grids)
        
        for dictionary in [self.grids, self.grid_names, self.grid_lengths]:
            if not self.are_all_values_unique(dictionary):
                raise ValueError("All values in dictionaries must be unique")

    def make_grid_dict(self, grid_names):
        grids = {}
        for idx_id, grid_name in grid_names.items():
            grid = getattr(self.model.par, grid_name, None)
            if grid is None:
                raise ValueError(f"Grid '{grid_name}' not found in model parameters")
            grids[idx_id] = grid
        return grids
            
    def make_grid_lengths_dict(self, grids):
        grid_lengths = {}
        for idx_id, grid in grids.items():
            grid_lengths[idx_id] = len(grid)
        return grid_lengths
        
            
    
    def are_all_values_unique(self, my_dict):
        """
        Checks if all values in the given dictionary are unique.
        Handles numpy.ndarray values by converting them to tuples for hashing.

        Args:
            my_dict (dict): The dictionary to check.

        Returns:
            bool: True if all values in the dictionary are unique, False otherwise.
        """
        seen_values = set()
        for value in my_dict.values():
            hashable_value = value
            if isinstance(value, np.ndarray):
                # Convert numpy array to a tuple to make it hashable
                # This works for arrays of any dimension
                hashable_value = tuple(value.flatten()) # Flatten and convert to tuple
            
            if hashable_value in seen_values:
                return False
            seen_values.add(hashable_value)
        return True
            
    
    # get key from value in dictionary
    def get_key_from_value(self, my_dict, target_value):
        """
        Retrieves the first key associated with a given value in a dictionary.

        Args:
            my_dict (dict): The dictionary to search within.
            target_value: The value to search for.

        Returns:
            The key associated with the target_value if found, otherwise None.
            If multiple keys have the same value, the key found first during
            iteration is returned.
        """
        for key, value in my_dict.items():
            if value == target_value:
                return key
        raise ValueError(f"Value {target_value} not found in dictionary")
    
    def get_variable_index_order(self, variable):
        return [self.get_key_from_value(self.grid_lengths, idx_length) for idx_length in variable.shape]


    def slice_variable(self, variable, index_slice: dict, grid_name):
        variable_index_order = self.get_variable_index_order(variable)
        
        # check that grid is in the variable_index_order
        grid_idx_id = self.get_key_from_value(self.grid_names, grid_name)
        if grid_idx_id not in variable_index_order:
            raise ValueError(f"Grid name '{grid_name}' not found in variable index")
        
        post_index_slice = [np.nan]*len(variable_index_order)
        for i, idx_id in enumerate(variable_index_order):
            if idx_id == grid_idx_id:
                post_index_slice[i] = slice(None, None, None) # use entire dimension, because we don't slice on the grid we are plotting over
            elif idx_id in index_slice.keys():
                post_index_slice[i] = index_slice[idx_id]
            else:
                print(f"WARNING: Index '{idx_id}' is not sliced upon")
        
        return tuple(post_index_slice)
    
    def make_default_label(self, variable, var_data, index, grid):
        my_label = ''
        for i, label_id in enumerate(self.labels):
            if label_id == 'variable':
                if i > 0:
                    my_label += ', '
                my_label += f"{variable}"
            elif label_id == 'model':
                if i > 0:
                    my_label += ', '
                my_label += f"{self.model_name}"
            elif label_id == 'index':
                if i > 0:
                    my_label += '\n'
                grid_modified = 'grid_A' if grid in {'grid_Aw', 'grid_Am'} else grid #OBS: This could be handled more generally
                variable_index_order = self.get_variable_index_order(var_data)
                label_indices = [
                    idx_id for idx_id in variable_index_order
                    if (idx_id in index) and (self.grid_names[idx_id] != grid_modified)
                ]
                label_texts = [
                    "{}[{}]={:.2f}".format(
                        self.grid_names[idx_id].removeprefix('grid_'),
                        index[idx_id],
                        self.grids[idx_id][index[idx_id]],
                    )
                    for idx_id in label_indices
                ]
                if label_texts:
                    label_text = ''
                    for j, label_idx in enumerate(label_texts):
                        label_text += label_idx
                        if j < len(label_texts) - 1:
                            label_text += ', '
                        if (j + 1) % self.index_labels_per_row == 0:
                            label_text += '\n'
                    my_label += label_text
            else:
                raise ValueError(f"Label '{label_id}' not recognized. Use 'variable', 'index', or 'model'.")
        return my_label
    
    def add_title(self, ax, variable):
        title = ''
        for i, title_id in enumerate(self.titles):
            if title_id == 'variable':
                if i > 0:
                    title += ', '
                title += f"{variable}"
            if title_id == 'model':
                if i > 0:
                    title += ', '
                title += f"{self.model_name}"
        ax.set_title(title)

    def plot_vars_over_grid(self, ax, variables, grid: str, index, namespace='sol', **kwargs):

        # check if ax has enough axes
        if len(ax) < len(variables):
            raise ValueError(f"length of ax ({len(ax)}) must be at least the number of variables ({len(variables)})")
    
        # unpack
        nmspc = getattr(self.model, namespace)
        if nmspc is None:
            raise ValueError(f"Namespace '{namespace}' not found in model")
        
        # get grid
        x = getattr(self.model.par, grid)
        if x is None:
            raise ValueError(f"Grid '{grid}' not found in model parameters")
        
        for i, var in enumerate(variables):
            # get variable
            if var is None:
                y = np.nan * np.ones_like(x)
            elif hasattr(nmspc, var):
                var_data = getattr(nmspc, var)
                grid_modified = 'grid_A' if grid in {'grid_Aw', 'grid_Am'} else grid
                idx_slice = self.slice_variable(var_data, index, grid_modified)
                y = var_data[idx_slice]
            else:
                raise ValueError(f"Variable '{var}' not found in namespace '{namespace}'")
            
            # handle label
            if var is None:
                label = None
            elif 'label' in kwargs and kwargs['label'] is not None:
                label = kwargs['label']
            else: # default label
                label = self.make_default_label(var, var_data, index, grid)
            
            # plot
            plot_kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
            ax[i].plot(x, y, label=label, **plot_kwargs)
            ax[i].set_xlabel(grid.removeprefix('grid_'))
            ax[i].legend()
            if self.titles:
                self.add_title(ax[i], var)
        
        return ax
    
    def plot_female_single_choices(self, ax, grid, index, **kwargs):
        variables = [
            'Cwd_tot_single_to_single',
            'lw_single_to_single',
            'Cwd_priv_single_to_single',
            'hwd_single_to_single',
            'Cwd_inter_single_to_single',
            'Qwd_single_to_single'
        ]
        if grid == 'grid_A':
            print("Warning: plotting over grid_A, which is not the same as grid_Aw or grid_Am. Make sure this is intended.")
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)

    def plot_male_single_choices(self, ax, grid, index, **kwargs):
        variables = [
            'Cmd_tot_single_to_single',
            'lm_single_to_single',
            'Cmd_priv_single_to_single',
            'hmd_single_to_single',
            'Cmd_inter_single_to_single',
            'Qmd_single_to_single'
        ]
        if grid == 'grid_A':
            print("Warning: plotting over grid_A, which is not the same as grid_Aw or grid_Am. Make sure this is intended.")
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)

    def plot_female_single_values(self, ax, grid, index, **kwargs):
        variables = [
            'Vwd_single_to_single',
            'Vw_single_to_single',
            'Vw_single_to_couple',
            'EVw_start_as_single',
            'EmargVw_start_as_single',
            'EVw_cond_meet_partner',
        ]
        if grid == 'grid_A':
            print("Warning: plotting over grid_A, which is not the same as grid_Aw or grid_Am. Make sure this is intended.")
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_male_single_values(self, ax, grid, index, **kwargs):
        variables = [
            'Vmd_single_to_single',
            'Vm_single_to_single',
            'Vm_single_to_couple',
            'EVm_start_as_single',
            'EmargVm_start_as_single',
            'EVm_cond_meet_partner',
        ]
        if grid == 'grid_A':
            print("Warning: plotting over grid_A, which is not the same as grid_Aw or grid_Am. Make sure this is intended.")
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_couple_choices(self, ax, grid, index, **kwargs):
        female_variables = [
            None,
            'lw_couple_to_couple',
            'Cwd_priv_couple_to_couple',
            'hwd_couple_to_couple',
            None,
            None,
        ]
        gridw = 'grid_Aw' if grid == 'grid_A' else grid
        self.plot_vars_over_grid(ax, female_variables, gridw, index, namespace='sol', **kwargs)
        
        male_variables = [
            None,
            'lm_couple_to_couple',
            'Cmd_priv_couple_to_couple',
            'hmd_couple_to_couple',
            None,
            None,
        ]
        gridm = 'grid_Am' if grid == 'grid_A' else grid
        self.plot_vars_over_grid(ax, male_variables, gridm, index, namespace='sol', **kwargs)
        
        total_variables = [
            'Cd_tot_couple_to_couple',
            None,
            None,
            None,
            'Cd_inter_couple_to_couple',
            'Qd_couple_to_couple',
        ]
        self.plot_vars_over_grid(ax, total_variables, grid, index, namespace='sol', **kwargs)
    
    def plot_female_couple_choices(self, ax, grid, index, **kwargs):
        variables = [
            'Cwd_priv_couple_to_couple',
            'lw_couple_to_couple',
            'hwd_couple_to_couple',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_male_couple_choices(self, ax, grid, index, **kwargs):
        variables = [
            'Cmd_priv_couple_to_couple',
            'lm_couple_to_couple',
            'hmd_couple_to_couple',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_gender_couple_choices(self, ax, grid, index, **kwargs):
        variables = [
            'Cwd_priv_couple_to_couple',
            'Cmd_priv_couple_to_couple',
            'lw_couple_to_couple',
            'lm_couple_to_couple',
            'hwd_couple_to_couple',
            'hmd_couple_to_couple',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_total_couple_choices(self, ax, grid, index, **kwargs):
        variables = [
            'Cd_tot_couple_to_couple',
            'Cd_inter_couple_to_couple',
            'Qd_couple_to_couple',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_female_couple_values(self, ax, grid, index, **kwargs):
        variables = [
            'Vw_start_as_couple',
            'EVw_start_as_couple',
            'Vw_couple_to_couple',
            'Vw_couple_to_single',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_male_couple_values(self, ax, grid, index, **kwargs):
        variables = [
            'Vm_start_as_couple',
            'EVm_start_as_couple',
            'Vm_couple_to_couple',
            'Vm_couple_to_single',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_total_couple_values(self, ax, grid, index, **kwargs):
        variables = [
            'Vd_couple_to_couple',
            'margV_start_as_couple',
            'EmargV_start_as_couple',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_female_single_pd(self, ax, grid, index, **kwargs):
        variables = [
            "EmargUwd_single_to_single_pd",
            "Cd_totw_single_to_single_pd",
            "Mwd_single_to_single_pd",
            "Vwd_single_to_single_pd",
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_male_single_pd(self, ax, grid, index, **kwargs):
        variables = [
            "EmargUmd_single_to_single_pd",
            "Cd_totm_single_to_single_pd",
            "Mmd_single_to_single_pd",
            "Vmd_single_to_single_pd",
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_couple_pd(self, ax, grid, index, **kwargs):
        variables = [
            'EmargUd_pd',
            'Cd_tot_pd',
            'Md_pd',
            'Vd_couple_to_couple_pd',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
    def plot_power(self, ax, grid, index, **kwargs):
        variables = [
            'power',
            'power_idx',
        ]
        return self.plot_vars_over_grid(ax, variables, grid, index, namespace='sol', **kwargs)
    
            
    def sim_to_df(self):

        par = self.model.par
        sim = self.model.sim

        # Extract variable names from sim that are arrays with shape (par.simN, par.simT)
        var_names = [
            key for key, val in sim.__dict__.items()
            if isinstance(val, np.ndarray) and val.shape == (par.simN, par.simT)
        ]

        data = {
            'id': np.repeat(np.arange(par.simN), par.simT),
            't': np.tile(np.arange(par.simT), par.simN)
        }

        for var in var_names:
            data[var] = getattr(sim, var).reshape(-1)

        df = pd.DataFrame(data)
        return df

    def plot_simulated_vars(self, ax, variables, where=None, agg_fct="mean", **kwargs):

        # check if ax has enough axes
        if len(ax) < len(variables):
            raise ValueError(f"length of ax ({len(ax)}) must be at least the number of variables ({len(variables)})")
    
        # Select data
        df = self.sim_data.copy()
        if where is not None:
            mask = df.eval(where)
            if not mask.any():
                raise ValueError(f"No data points match the condition: {where}")
            df = df[mask]

        # Get time grid
        x = np.arange(self.model.par.simT)

        for i, var in enumerate(variables):
            if var is None:
                y = np.full_like(x, np.nan, dtype=float)
            elif var in df.columns:
                var_data = df[['id', 't', var]].sort_values(['t', 'id'])
                if agg_fct:
                    y = var_data.groupby('t')[var].agg(agg_fct).reindex(x, fill_value=np.nan).values
                else:
                    y = var_data.pivot(index='t', columns='id', values=var).reindex(index=x).values
            else:
                raise ValueError(f"Variable '{var}' not found in simulated data")
            # handle label
            if var is None:
                label = None
            elif 'label' in kwargs and kwargs['label'] is not None:
                label = kwargs['label']
            elif agg_fct: # default label with aggregation function
                label = f"{agg_fct.capitalize() if isinstance(agg_fct, str) else agg_fct.__name__.capitalize()} of {var} ({self.model_name})"
            else: # default label
                label = None
            
            # plot
            plot_kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
            ax[i].plot(x, y, label=label, **plot_kwargs)
            ax[i].set_xlabel("Period (t)")
            ax[i].set_xlim([x.min()-0.5, x.max()+0.5])
            if label:
                ax[i].legend()
            if self.titles:
                self.add_title(ax[i], var)
            if agg_fct is None:
                # Set all lines to grey, alpha=0.1, and no markers
                for line in ax[i].get_lines():
                    if 'color' in kwargs and kwargs['color'] is not None:
                        line.set_color(kwargs['color'])
                    else:
                        line.set_color('grey')
                    line.set_linestyle('-')
                    line.set_alpha(0.1)
                    line.set_marker('')
        
        return ax
    
    def plot_sim_female_single(self, ax, agg_fct='mean', **kwargs):
        singles_mask = 'couple == 0'
        variables = ['Aw','Cw_tot','Cw_priv', 'Cw_inter', 'lw','hw','Qw','util']
        return self.plot_simulated_vars(ax, variables, where=singles_mask, agg_fct=agg_fct, **kwargs)
    
    def plot_sim_male_single(self, ax, agg_fct='mean', **kwargs):
        singles_mask = 'couple == 0'
        variables = ['Am','Cm_tot','Cm_priv', 'Cm_inter', 'lm','hm','Qm']
        return self.plot_simulated_vars(ax, variables, where=singles_mask, agg_fct=agg_fct, **kwargs)
        
    def plot_sim_female_couple(self, ax, agg_fct='mean', **kwargs):
        couples_mask = 'couple == 1'
        variables = ['A','Cw_tot','Cw_priv', 'Cw_inter', 'lw','hw','Qw','util']
        return self.plot_simulated_vars(ax, variables, where=couples_mask, agg_fct=agg_fct, **kwargs)

    def plot_sim_male_couple(self, ax, agg_fct='mean', **kwargs):
        couples_mask = 'couple == 1'
        variables = ['A','Cm_tot','Cm_priv', 'Cm_inter', 'lm','hm','Qm']
        return self.plot_simulated_vars(ax, variables, where=couples_mask, agg_fct=agg_fct, **kwargs)

    def plot_sim_couple(self, ax, agg_fct='mean', **kwargs):
        couples_mask = 'couple == 1'
        variables = ['A','C_tot', 'power', 'love']
        return self.plot_simulated_vars(ax, variables, where=couples_mask, agg_fct=agg_fct, **kwargs)

    def plot_sim_female(self, ax, where=None, agg_fct='mean', **kwargs):
        variables = ['Aw','Cw_tot','Cw_priv', 'Cw_inter', 'lw','hw','Qw','couple']
        return self.plot_simulated_vars(ax, variables, where=where, agg_fct=agg_fct, **kwargs)

    def plot_sim_male(self, ax, where=None, agg_fct='mean', **kwargs):
        variables = ['Am','Cm_tot','Cm_priv', 'Cm_inter', 'lm','hm','Qm', 'couple']
        return self.plot_simulated_vars(ax, variables, where=where, agg_fct=agg_fct, **kwargs)
