#Imports
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ccmodels.analysis.utils import zero_center_and_shift, untuned_shifter


def bootstrap_medians(x):
    '''Small utility function to compute standard error of the median'''
    medians_b = []
    for i in range(1000):
        samp = np.random.choice(x.values, x.shape[0], replace = True)
        medians_b.append(np.median(samp))
    
    return np.std(medians_b)


def tpo_po_simulator_new(n_neurons, Nl23, Nl4, l4_pre_tuned, l23_pre_tuned, 
                         l4_pre_untuned, l23_pre_untuned, weighted = True, pre_profile = 'both'):
    '''
    This function calculates the difference between the set Preferred orientation (tPO) and the preferred orientation
    observed by the current generated by the simulated presynaptic network (oPO). It does so by also including information
    on UNTUNED presynaptic neurons and by centering the presynaptic tuning curve at 0 so as to then be able to 
    shift it by a drawn delta ori value, effectively simulating an entirely new presynaptic tuning curve for the 
    postsynatic neuron
    
    
    Parameters:
    n_neurons: number of postsynaptic neurons to model
    Nl23: number of L23 presynaptic inputs
    Nl4: number of l4 presynaptic inputs
    l4_pre_tuned: df with the activities and infromation on each TUNED presynaptic neuron of L4
    l23_pre_tuned: df with the activities and infromation on each TUNED presynaptic neuron of L23
    l4_pre_untuned: df with the activities and infromation on each UNTUNED presynaptic neuron of L4
    l23_pre_untuned: df with the activities and infromation on each UNTUNED presynaptic neuron of L23
    weighted: bool, whether to multiply by connection strength or not 
    pre_profile: str, the presynaptic profile of the simulated neurons, can be either 'tuned' for only selective preynspatic neurons
    'untuned' for nonselective presynaptic neurons or 'both' for both thepresence of both types
    
    Returns: 
    diff_po: list, differences between tPO and oPO for all simulated postsynaptic neurons    
    
    '''
    pd.options.mode.chained_assignment = None  # default='warn'

    #np.random.seed(4)
    diff_po = [] #store the differences in PO
    #pref_ors_post = []
    #converged_pos = []
    #simulated_currents = np.zeros(16)
    if pre_profile == 'both':
        nl23tuned = int(round(Nl23*0.55,0))
        nl23untuned = int(round(Nl23*0.45,0))

        nl4tuned = int(round(Nl4*0.55,0))
        nl4untuned = int(round(Nl4*0.45,0))
    else:
        nl23tuned = Nl23
        nl4tuned = Nl4

        nl23untuned = Nl23
        nl4untuned = Nl4

    dirssamp = list(set(l4_pre_tuned['delta_ori_constrained'].round(3)))
    
    for i in tqdm(range(n_neurons)):
        post_po = 0

        #Storage of summed weighted activities
        current_sim = np.zeros(16)


        #Extract current of TUNED inputs 
        #L4

        #Extract subset of delta ori (this is to follow connection probability dist)
        ori_drawsl4_tuned = np.random.choice(l4_pre_tuned['delta_ori_constrained'],nl4tuned)
        
        #Extract random sample of neuronal responses
        subset_l4_tuned = l4_pre_tuned.sample(nl4tuned)
        subset_l4_tuned['delta_ori_sim'] = list(ori_drawsl4_tuned)

        #Center tuning curves at 0 and shift them by the extracted delta ori
        reordered_act_l4tuned, constrained_dirs = zero_center_and_shift(subset_l4_tuned, 'pre_orientations', 
                                                                'pre_po', ori_drawsl4_tuned, currents = False)
        

        #L2/3
        ori_drawsl23_tuned = np.random.choice(l23_pre_tuned['delta_ori_constrained'],nl23tuned)

        subset_l23_tuned = l23_pre_tuned.sample(nl23tuned)
        subset_l23_tuned['delta_ori_sim'] = list(ori_drawsl23_tuned)

        reordered_act_l23tuned, constrained_dirs = zero_center_and_shift(subset_l23_tuned, 'pre_orientations', 
                                                                'pre_po', ori_drawsl23_tuned, currents = False)
        

        #Extract the wights randomly 
        strength_l4_tuned = np.array(list(np.random.choice(l4_pre_tuned['size'],nl4tuned)))
        strength_l23_tuned = np.array(list(np.random.choice(l23_pre_tuned['size'], nl23tuned)))

        

        #Extract current of UNTUNED inputs 
        #L4
        #Draw random sample of untuned neurons
        subset_l4_untuned = l4_pre_untuned.sample(nl4untuned, replace = True)
        ori_drawsl4_untuned = np.random.choice(dirssamp,nl4untuned) #[0]*nl4tuned #np.random.choice(dirssamp,nl4tuned)
        
        #Constrain them in -pi, pi range and shift them by a random value
        reordered_act_l4untuned, constrained_dirs = untuned_shifter(subset_l4_untuned, 'pre_orientations', ori_drawsl4_untuned, currents = False)            

        #L2/3
        subset_l23_untuned = l23_pre_untuned.sample(nl23untuned, replace = True)
        ori_drawsl23_untuned = np.random.choice(dirssamp,nl23untuned) #[0]*nl4tuned #np.random.choice(dirssamp,nl4tuned)

        #Constrain them in -pi, pi range and shift them by a random value
        reordered_act_l23untuned, constrained_dirs = untuned_shifter(subset_l23_untuned, 'pre_orientations', ori_drawsl23_untuned, currents = False)
            

        #Sample set of weights
        strength_l4_untuned = np.array(list(np.random.choice(l4_pre_untuned['size'],nl4untuned)))
        strength_l23_untuned = np.array(list(np.random.choice(l23_pre_untuned['size'], nl23untuned)))

        

        if pre_profile ==  'tuned':
            #Choose whether to inlcude them in activity
            if weighted:   
                #Calculate the total current and sum it
                current_sim+= np.sum(np.array(reordered_act_l4tuned)*strength_l4_tuned.reshape((nl4tuned,1)), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23tuned)*strength_l23_tuned.reshape(nl23tuned,1), axis = 0)
            else:
                current_sim+= np.sum(np.array(reordered_act_l4tuned), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23tuned), axis = 0)

        elif pre_profile== 'untuned':
            if weighted:   
                #Calculate the total current and sum it
                current_sim+= np.sum(np.array(reordered_act_l4untuned)*strength_l4_untuned.reshape((nl4untuned,1)), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23untuned)*strength_l23_untuned.reshape(nl23untuned,1), axis = 0)
            else:
                current_sim+= np.sum(np.array(reordered_act_l4untuned), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23untuned), axis = 0)
        
        elif pre_profile == 'both':
            if weighted:   
                #Calculate the total current and sum it
                current_sim+= np.sum(np.array(reordered_act_l4tuned)*strength_l4_tuned.reshape((nl4tuned,1)), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23tuned)*strength_l23_tuned.reshape(nl23tuned,1), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l4untuned)*strength_l4_untuned.reshape((nl4untuned,1)), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23untuned)*strength_l23_untuned.reshape(nl23untuned,1), axis = 0)
            else:
                current_sim+= np.sum(np.array(reordered_act_l4tuned), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23tuned), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l4untuned), axis = 0)
                current_sim+= np.sum(np.array(reordered_act_l23untuned), axis = 0)
        else:
            raise('Mode not recoginzed, should be one of either tuned, untuned or both')

        #Find preferred orientatio of simulated postsynaptic current
        current_po = l4_pre_tuned['new_dirs'].values[0][np.argmax(current_sim)]

        #Substract it to the observed postsynaptic current
        diff_po.append(float(current_po-post_po))

    return diff_po


def input_current_simulator(n_neurons, Nl23, Nl4, l4_pre_tuned, l23_pre_tuned, 
                         l4_pre_untuned, l23_pre_untuned, weighted = True, pre_profile = 'both'):
    '''
    This function calculates the difference between the set Preferred orientation (tPO) and the preferred orientation
    observed by the current generated by the simulated presynaptic network (oPO). It does so by also including information
    on UNTUNED presynaptic neurons and by centering the presynaptic tuning curve at 0 so as to then be able to 
    shift it by a drawn delta ori value, effectively simulating an entirely new presynaptic tuning curve for the 
    postsynatic neuron
    
    
    Parameters:
    n_neurons: number of postsynaptic neurons to model
    Nl23: number of L23 presynaptic inputs
    Nl4: number of l4 presynaptic inputs
    l4_pre_tuned: df with the activities and infromation on each TUNED presynaptic neuron of L4
    l23_pre_tuned: df with the activities and infromation on each TUNED presynaptic neuron of L23
    l4_pre_untuned: df with the activities and infromation on each UNTUNED presynaptic neuron of L4
    l23_pre_untuned: df with the activities and infromation on each UNTUNED presynaptic neuron of L23
    weighted: bool, whether to multiply by connection strength or not 
    pre_profile: str, the presynaptic profile of the simulated neurons, can be either 'tuned' for only selective preynspatic neurons
    'untuned' for nonselective presynaptic neurons or 'both' for both thepresence of both types
    
    Returns: 
    diff_po: list, differences between tPO and oPO for all simulated postsynaptic neurons    
    
    '''
    pd.options.mode.chained_assignment = None  # default='warn'

    #np.random.seed(4)    

    if pre_profile == 'both':
        nl23tuned = int(round(Nl23*0.55,0))
        nl23untuned = int(round(Nl23*0.45,0))

        nl4tuned = int(round(Nl4*0.55,0))
        nl4untuned = int(round(Nl4*0.45,0))
    else:
        nl23tuned = Nl23
        nl4tuned = Nl4

        nl23untuned = Nl23
        nl4untuned = Nl4

    dirssamp = list(set(l4_pre_tuned['delta_ori_constrained'].round(3)))
    
    current_l4 = []
    current_l23 = []

    for i in tqdm(range(n_neurons)):
        post_po = 0

        #Storage of summed weighted activities
        current_sim = np.zeros(16)


        #Extract current of TUNED inputs 
        #L4

        #Extract subset of delta ori (this is to follow connection probability dist)
        ori_drawsl4_tuned = np.random.choice(l4_pre_tuned['delta_ori_constrained'],nl4tuned)
        
        #Extract random sample of neuronal responses
        subset_l4_tuned = l4_pre_tuned.sample(nl4tuned)
        subset_l4_tuned['delta_ori_sim'] = list(ori_drawsl4_tuned)

        #Center tuning curves at 0 and shift them by the extracted delta ori
        reordered_act_l4tuned, constrained_dirs = zero_center_and_shift(subset_l4_tuned, 'pre_orientations', 
                                                                'pre_po', ori_drawsl4_tuned, currents = False)
        

        #L2/3
        ori_drawsl23_tuned = np.random.choice(l23_pre_tuned['delta_ori_constrained'],nl23tuned)

        subset_l23_tuned = l23_pre_tuned.sample(nl23tuned)
        subset_l23_tuned['delta_ori_sim'] = list(ori_drawsl23_tuned)

        reordered_act_l23tuned, constrained_dirs = zero_center_and_shift(subset_l23_tuned, 'pre_orientations', 
                                                                'pre_po', ori_drawsl23_tuned, currents = False)
        

        #Extract the wights randomly 
        strength_l4_tuned = np.array(list(np.random.choice(l4_pre_tuned['size'],nl4tuned)))
        strength_l23_tuned = np.array(list(np.random.choice(l23_pre_tuned['size'], nl23tuned)))

        

        #Extract current of UNTUNED inputs 
        #L4
        #Draw random sample of untuned neurons
        subset_l4_untuned = l4_pre_untuned.sample(nl4untuned, replace = True)
        ori_drawsl4_untuned = np.random.choice(dirssamp,nl4untuned) #[0]*nl4tuned #np.random.choice(dirssamp,nl4tuned)
        
        #Constrain them in -pi, pi range and shift them by a random value
        reordered_act_l4untuned, constrained_dirs = untuned_shifter(subset_l4_untuned, 'pre_orientations', ori_drawsl4_untuned, currents = False)            

        #L2/3
        subset_l23_untuned = l23_pre_untuned.sample(nl23untuned, replace = True)
        ori_drawsl23_untuned = np.random.choice(dirssamp,nl23untuned) #[0]*nl4tuned #np.random.choice(dirssamp,nl4tuned)

        #Constrain them in -pi, pi range and shift them by a random value
        reordered_act_l23untuned, constrained_dirs = untuned_shifter(subset_l23_untuned, 'pre_orientations', ori_drawsl23_untuned, currents = False)
            

        #Sample set of weights
        strength_l4_untuned = np.array(list(np.random.choice(l4_pre_untuned['size'],nl4untuned)))
        strength_l23_untuned = np.array(list(np.random.choice(l23_pre_untuned['size'], nl23untuned)))

        

        if pre_profile ==  'tuned':
            #Choose whether to inlcude them in activity
            if weighted:   
                #Calculate the total current and sum it
                current_sim_l4 = np.array(reordered_act_l4tuned)*strength_l4_tuned.reshape(nl4tuned,1)
                current_sim_l23= np.array(reordered_act_l23tuned)*strength_l23_tuned.reshape(nl23tuned,1)
                if len(current_l23) == 0:
                    current_l4 = current_sim_l4
                    current_l23 = current_sim_l23
                else:
                    current_l4 = np.vstack((current_l4, current_sim_l4))
                    current_l23 = np.vstack((current_l23,current_sim_l23))
            else:
                current_sim_l4= np.array(reordered_act_l4tuned)
                current_sim_l23 = np.array(reordered_act_l23tuned)

                current_l4 = np.vstack((current_l4, current_sim_l4))
                current_l23 = np.vstack((current_l23,current_sim_l23))

        elif pre_profile== 'untuned':
            if weighted:   
                #Calculate the total current and sum it
                current_sim_l4= np.array(reordered_act_l4untuned)*strength_l4_untuned.reshape(nl4untuned,1)
                current_sim_l23= np.array(reordered_act_l23untuned)*strength_l23_untuned.reshape(nl23untuned,1)

                current_l4 = np.vstack((current_l4, current_sim_l4))
                current_l23 = np.vstack((current_l23,current_sim_l23))
            else:
                current_sim_l4= np.array(reordered_act_l4untuned)
                current_sim_l23= np.array(reordered_act_l23untuned)

                current_l4 = np.vstack((current_l4, current_sim_l4))
                current_l23 = np.vstack((current_l23,current_sim_l23))
        
        elif pre_profile == 'both':
            if weighted:   
                #Calculate the total current and sum it
                current_sim_l4= np.array(reordered_act_l4tuned)*strength_l4_tuned.reshape(nl4tuned,1)
                current_sim_l23= np.array(reordered_act_l23tuned)*strength_l23_tuned.reshape(nl23tuned,1)
                current_sim_l4 = np.vstack(current_sim_l4, np.array(reordered_act_l4untuned)*strength_l4_untuned.reshape(nl4untuned,1))
                current_sim_l23 = np.vstack(current_sim_l23, np.array(reordered_act_l23untuned)*strength_l23_untuned.reshape(nl23untuned,1))

                current_l4 = np.vstack((current_l4, current_sim_l4))
                current_l23 = np.vstack((current_l23,current_sim_l23))
            else:
                current_sim_l4= np.array(reordered_act_l4tuned)
                current_sim_l23= np.array(reordered_act_l23tuned)
                current_sim_l4 = np.vstack((current_sim_l4, np.array(reordered_act_l4untuned)))
                current_sim_l23 = np.vstack((current_sim_l23, np.array(reordered_act_l23untuned)))

                current_l4 = np.vstack((current_l4, current_sim_l4))
                current_l23 = np.vstack((current_l23,current_sim_l23))
        else:
            raise('Mode not recoginzed, should be one of either tuned, untuned or both')

    return current_l4, current_l23


def bootstrap_conn_prob(connectome_subset, 
                        pre_layer: str,
                        half_dirs: bool = False ):
    '''calculates boostrap mean and standard error for connection porbability for presynpatic neurons
    for a specific layer as a function of the difference in preferred orientation
    
    half_dirs: if true directions between 0 and pi else between 0 and pi
    '''

    ##### Connection probability for L4 -> L2/3 ######
    tuned_neurons = connectome_subset[connectome_subset['pre_layer'] == pre_layer]

    #Calculate the sum of how many connections are in each difference group
    grouped_diffs = tuned_neurons.groupby('delta_ori_constrained')['post_id'].count().reset_index()


    #Rename columns and generate connection porbability
    grouped_diffs = grouped_diffs.rename(columns = {'post_id':'n_connections'})
    grouped_diffs['prob_connection']=grouped_diffs['n_connections']/np.sum(grouped_diffs['n_connections'])


    #Bootstrap
    np.random.seed(4)
    n_samps = 1000
    resamples = np.zeros((n_samps, grouped_diffs.shape[0]))
    dirs = np.zeros((n_samps, grouped_diffs.shape[0]))

    for i in range(n_samps):

        boot_samp = tuned_neurons.sample(frac = 1, replace = True)


        #Calculate the sum of how many connections are in each difference group
        boot_diffs = boot_samp.groupby('delta_ori_constrained')['post_id'].count().reset_index()


        #Rename columns and generate connection porbability
        boot_diffs = boot_diffs.rename(columns = {'post_id':'n_connections'})
        boot_diffs['prob_connection']=boot_diffs['n_connections']/np.sum(boot_diffs['n_connections'])
        resamples[i, :] = boot_diffs['prob_connection'].values
        dirs[i, :] = boot_diffs['delta_ori_constrained']

    if half_dirs:
        dirs = np.abs(dirs)
    else: 
        dirs = dirs

    boots_data = pd.DataFrame({'directions': dirs.ravel(), 'probs': resamples.ravel()})
    boots_data_std = boots_data.groupby('directions').std().reset_index()['probs']
    boots_data_mean = boots_data.groupby('directions').mean().reset_index()

    boots_data_clean = pd.DataFrame({'directions':boots_data_mean['directions'],'mean':boots_data_mean['probs'], 
                                     'std':boots_data_std})

    return boots_data_clean


def bootstrap_layerinput_proportions(data, layer_column, counts_column, layer_labels = None, n_iters = 100):
   
    # dists_matrix = []

    if layer_labels == None:
        iter_labels = sorted(list(set(data[layer_column].values)))
    else:
        iter_labels = layer_labels

    bootstrap_matrix = np.zeros((len(iter_labels), n_iters))

    for iter in tqdm(range(n_iters)):

        for layer_id in range(len(iter_labels)):
            layer_dist = data[data[layer_column] == iter_labels[layer_id]][counts_column].values
            layer_sample = np.random.choice(layer_dist, layer_dist.shape[0])
            bootstrap_matrix[layer_id, iter] = np.mean(layer_sample)

        bootstrap_matrix[:, iter] = bootstrap_matrix[:, iter]/np.sum(bootstrap_matrix[:, iter])

    bootstrap_samples = pd.DataFrame(columns = iter_labels, data =bootstrap_matrix.T)

    return bootstrap_samples