import math
import os
import itertools
import numpy as n
import time
import pickle
import sys
import scipy
import scipy.optimize as optimize
from multiprocessing import Pool, cpu_count

total_time = time.time()

#Defining Global Constants to use
ec = 4.8E-10
ev_to_erg = 1.6021773E-12

EAO_1 = -1.461 * ev_to_erg 
EAO_2 = 7.72 * ev_to_erg 
EAO_3 = 15 * ev_to_erg 
IPO_1 = 13.61806 * ev_to_erg
IPO_2 = 35.117 * ev_to_erg 

IPCU_1 = 7.72 * ev_to_erg 
IPCU_2 = 20.29 * ev_to_erg
IPCU_3 = 36.841 * ev_to_erg
IPCU_4 = 57.38 * ev_to_erg 
IPCU_5 = 79.8 * ev_to_erg 


#%% Polarization Calculation Details

#Function to add the right Ionization Potentials/Electron Affinities to a bond O and Cu sites depending on the signs of the charge induced by another bond
def EA_IP_func(q, pos, hole_positions_array):

    #Determining if the site the charge sits on is O or Cu
    site_type = 0 if int(pos[0] + pos[1]) != (pos[0] + pos[1]) else 1
    
    #Checking if this site already has doped holes
    if hole_positions_array.count(pos) == 0:
        if site_type == 0:
            if q >= 0: 
                return EAO_2 * abs(q/ec) - EAO_1 * abs(q/ec)
                
            else:
                return EAO_2 * abs(q/ec) - EAO_3 * abs(q/ec)
                
        if site_type == 1:
            if q >= 0: 
                return IPCU_2 * abs(q/ec) - IPCU_3 * abs(q/ec)
            else:
                return IPCU_2 * abs(q/ec) - IPCU_1 * abs(q/ec)
            
    elif hole_positions_array.count(pos) == 1:
        if site_type == 0:
            if q >= 0: 
                return -EAO_1 * (1-abs(q/ec)) + IPO_1 * abs(q/ec) + EAO_2
            else:
                return -EAO_1 * (1-abs(q/ec)) - EAO_2 * abs(q/ec) + EAO_2
            
        if site_type == 1:
            if q >= 0: 
                return -IPCU_3 * (1-abs(q/ec)) - IPCU_4 * abs(q/ec) + IPCU_2
            else:
                return -IPCU_3 * (1-abs(q/ec)) - IPCU_2 * abs(q/ec) + IPCU_2
        
        
    elif hole_positions_array.count(pos) == 2:
        if site_type == 0:
            if q >= 0: 
                return IPO_1 * (1-abs(q/ec)) + IPO_2 * abs(q/ec) + EAO_2
            else:
                return IPO_1 * (1-abs(q/ec)) - EAO_1 * abs(q/ec) + EAO_2
            
        if site_type == 1:
            if q >= 0: 
                return -IPCU_4 * (1-abs(q/ec)) - IPCU_5 * abs(q/ec) + IPCU_2
            else:
                return -IPCU_4 * (1-abs(q/ec)) - IPCU_3 * abs(q/ec) + IPCU_2
            
    else:
        print("More than 2 dopes holes on same site???")
        return 0


 
#Function to create the Cu-4O cluster Hamiltonian, diagonalize it numerically and return the component corresponding to the right bond of the GS eigenstate
def H_bonds(tdp, tpp, delta_up_right, delta_up_left, delta_down_left, delta_down_right):
    dim = 5
    H = n.array([ [0, -tdp, -tdp, tdp, tdp], [-tdp,delta_up_right,tpp,0,-tpp], [-tdp,tpp,delta_up_left,-tpp,0], [tdp,0,-tpp,delta_down_left,tpp], [tdp,-tpp,0,tpp,delta_down_right] ])
    
    #diagonalizing the Hamiltonian and populating the evecs and evals arrays
    evals,evecs = n.linalg.eigh(H)
    
    GS_index = n.argmin(evals)
    
    return evecs[:,GS_index]**2


#Single hole Cu-O dipoles calculated based on charge transfer energy difference
def cluster_energy_CuO(first_hole_pos, second_hole_pos, a, polarizability_O, tdp, tpp, delta_0, r_0, hole_config):
    
    start_time = time.time()

    #Defining the all_positions_array and the hole_positions_array from input hole position
    hole_positions_array = [first_hole_pos, second_hole_pos]
    defect_dist = int(math.ceil(n.linalg.norm(n.subtract(second_hole_pos, first_hole_pos)))) + 1
    cluster_size = r_0 + defect_dist
    unconstrained_positions = itertools.product(n.linspace(-cluster_size,cluster_size,2*cluster_size+1), n.linspace(-cluster_size,cluster_size,2*cluster_size+1))

    #Converting energy constants from eV to erg
    tdp = tdp * ev_to_erg
    tpp = tpp * ev_to_erg
    delta_0 = delta_0 * ev_to_erg
    VM_bare = (delta_0 + EAO_2 + IPCU_2)/2
  
    #Positions of O and Cu sites from origin, along with a cluster array with entries [x_up_pos, x_left_pos, x_down_pos, x_right_pos] for the index corresponding to a x_Cu entry
    x_Cu = [list(i) for i in unconstrained_positions if n.linalg.norm(n.subtract(i,first_hole_pos)) <= r_0 or n.linalg.norm(n.subtract(i,second_hole_pos)) <= r_0]
    x_O_up = [ n.add(Cu_pos,[0,0.5]).tolist() for Cu_pos in x_Cu ]
    x_O_left = [ n.add(Cu_pos,[-0.5,0]).tolist() for Cu_pos in x_Cu ]
    x_O_down = [ n.add(Cu_pos,[0,-0.5]).tolist() for Cu_pos in x_Cu ]
    x_O_right = [ n.add(Cu_pos,[0.5,0]).tolist() for Cu_pos in x_Cu ]
    x_O = n.unique(n.concatenate((x_O_up, x_O_left, x_O_down, x_O_right), axis=0), axis=0).tolist()
    x_cluster = [ [Cu_pos, O_up_pos, O_left_pos, O_down_pos, O_right_pos] for (Cu_pos, O_up_pos, O_left_pos, O_down_pos, O_right_pos) in zip(x_Cu, x_O_up, x_O_left, x_O_down, x_O_right)]
    
    
    #Modifying the hole_positions_array depending on which holes I want to include // hole_config = {0: no holes, 1: both holes, 2: Only First Hole, 3: Only Second Hole }
    if hole_config == 0:
        hole_positions_array = ()
    elif hole_config == 2:
        hole_positions_array = [hole_positions_array[0]]
    elif hole_config == 3:
        hole_positions_array = [hole_positions_array[1]]
    
    defect_mono_charges = [[ec,hole_pos] for hole_pos in hole_positions_array]
    
    
    #### Initial Guess Charges
    
    ### Potential Differences for all Cu-4O clusters
    potential_difference_clusters = n.zeros((len(x_cluster),4))
    #Looping over the 4 bonds of the cluster (excluding first entry of cluster list since that is the Cu pos)
    for cluster_idx, cluster in enumerate(x_cluster):
        Cu_pos = cluster[0]
        for O_idx, O_pos in enumerate(cluster[1:]):
            
            #Looping over the holes
            for hole_q, hole_pos in defect_mono_charges:
                
                defect_to_O_dist = n.linalg.norm(n.subtract(O_pos, hole_pos)) * a
                defect_to_Cu_dist = n.linalg.norm(n.subtract(Cu_pos, hole_pos)) * a
                
                #Potential on O (defect on different site)
                if defect_to_O_dist > 0:
                    potential_difference_clusters[cluster_idx][O_idx] += hole_q * 1/defect_to_O_dist
    
                #Potential on Cu (defect on different site)
                if defect_to_Cu_dist > 0:
                    potential_difference_clusters[cluster_idx][O_idx] += hole_q * -1/defect_to_Cu_dist
    
                    
            #Potential on O (defect(s) on same site)
            if hole_positions_array.count(O_pos) == 1:
                potential_difference_clusters[cluster_idx][O_idx] += (EAO_2 - EAO_1)/ec
            elif hole_positions_array.count(O_pos) == 2:
                potential_difference_clusters[cluster_idx][O_idx] += (EAO_2 + IPO_1)/ec  
    
                
            #Potential on Cu (defect(s) on same site)
            if hole_positions_array.count(Cu_pos) == 1:
                potential_difference_clusters[cluster_idx][O_idx] += (IPCU_2 - IPCU_3)/ec
            elif hole_positions_array.count(Cu_pos) == 2:
                potential_difference_clusters[cluster_idx][O_idx] += (IPCU_2 - IPCU_4)/ec 


    #Array of initial delta_0 + delta' and Evecs Squared for all clusters
    delta_clusters_initial = n.array([delta_0 + (ec) * pot_diff for pot_diff in potential_difference_clusters])
    evecs_clusters_initial =  n.array([H_bonds(tdp, tpp, *delta_cluster) for delta_cluster in delta_clusters_initial] )
    
       
    #Initial Guesses for O and Cu effective charges            
    initial_guess_Cu = n.array([ec*evec[0] for evec in  evecs_clusters_initial] )
    initial_guess_O = n.zeros(len(x_O))
    for O_idx, O_pos in enumerate(x_O):
        
        #Determining which clusters contains the O
        clusters = [cluster for cluster in x_cluster if O_pos in cluster ]
        cluster_indices = [x_cluster.index(cluster) for cluster in clusters]
        bond_indices = [cluster.index(O_pos) for cluster in clusters]    
        for (cluster_idx, bond_idx) in zip(cluster_indices,bond_indices):
            initial_guess_O[O_idx] += ec*evecs_clusters_initial[cluster_idx][bond_idx]
                
            
    #### Initial Guess Dipoles
    initial_monopole_field_at_dipoles = n.zeros((len(x_O),2))
    for dip_idx,dip_pos in enumerate(x_O):
        initial_monopole_field_at_dipoles[dip_idx] = n.sum( [ def_q*(n.subtract(dip_pos,def_pos)*a )/(n.linalg.norm(n.subtract(dip_pos,def_pos))*a)**3 if n.linalg.norm(n.subtract(dip_pos,def_pos)) > 0 else n.array([0.0,0.0]) for (def_q,def_pos) in defect_mono_charges ], axis=0).tolist()
        
    initial_guess_O_dipoles = [(polarizability_O * 1E-24 * field).tolist() for field in initial_monopole_field_at_dipoles]
    
    
    #Combining the initial guesses into an array with order [ O_dipoles, O_charges, Cu_charges]
    initial_guess = n.concatenate((n.hstack(initial_guess_O_dipoles), initial_guess_O, initial_guess_Cu))
    
    
    
    #Initializing the lists of strings representing the equations to solve 
    #Equation Order: [0 -> 2len(x_0]: O dipoles // [2len(x_O) -> 3len(x_O)]: O induced charges // [3len(x_O) -> 3len(x_O) + len(x_Cu)]: Cu induced charges
    equation_string_array = ["-x["+ str(i) + "]" for i in range(2*len(x_O) + len(x_O) + len(x_Cu))]
    delta_clusters = [ [str(delta) for delta in delta_clusters_initial[cluster_idx]] for cluster_idx,_ in enumerate(x_cluster)]
    
    #return delta_clusters
    #### O dipoles Equations
    for O_idx, O_pos in enumerate(x_O):
        
        #Hole influence on O dipoles
        O_equation_string_x =  "+" + str(polarizability_O * (1E-24) * initial_monopole_field_at_dipoles[O_idx][0])
        O_equation_string_y =  "+" + str(polarizability_O * (1E-24) * initial_monopole_field_at_dipoles[O_idx][1])
 
        ###  O (dipoles and charges)  -->  O dipoles
        for O_sum_idx, O_sum_pos in enumerate(x_O):
            
            O_to_O_vec = n.subtract(O_pos, O_sum_pos)*a
            O_to_O_dist = n.linalg.norm(O_to_O_vec)

            ### O dipoles --> O dipoles
            if O_to_O_dist > 0:
                O_equation_string_x += " + x[" + str(2*O_sum_idx) + "] * " + str( polarizability_O*(1E-24)*(3*(O_to_O_vec[0]**2)*(1/O_to_O_dist)**5 - (1/O_to_O_dist)**3) )
                O_equation_string_x += " + x[" + str(2*O_sum_idx+1) + "] * " +  str(polarizability_O*(1E-24)*(3*(O_to_O_vec[0]*O_to_O_vec[1])*(1/O_to_O_dist)**5))
                O_equation_string_y += " + x[" + str(2*O_sum_idx) + "] * " + str(polarizability_O*(1E-24)*(3*(O_to_O_vec[0]*O_to_O_vec[1])*(1/O_to_O_dist)**5))
                O_equation_string_y += " + x[" + str(2*O_sum_idx+1) + "] * " + str(polarizability_O*(1E-24)*(3*(O_to_O_vec[1]**2)*(1/O_to_O_dist)**5 - (1/O_to_O_dist)**3))
        
 
            ### O charges --> O dipoles
            if O_to_O_dist > 0:
                O_equation_string_x += " + x[" + str(O_sum_idx + 2*len(x_O)) + "] *" + str(  polarizability_O*(1E-24)* ( O_to_O_vec[0] /(O_to_O_dist**3) )  )
                O_equation_string_y += " + x[" + str(O_sum_idx + 2*len(x_O)) + "] *" + str(  polarizability_O*(1E-24)* ( O_to_O_vec[1] /(O_to_O_dist**3) )  )
                
 
    
        ###  Cu charges --> O dipoles
        for Cu_sum_idx, Cu_sum_pos in enumerate(x_Cu):   
            
            Cu_to_O_vec = n.subtract(O_pos, Cu_sum_pos)*a
            Cu_to_O_dist = n.linalg.norm(Cu_to_O_vec)
            
            #Cu charges --> O dipoles
            O_equation_string_x += " + x[" + str(Cu_sum_idx + 3*len(x_O)) + "] *" + str(  polarizability_O*(1E-24)* ( Cu_to_O_vec[0] /(Cu_to_O_dist**3) )  )
            O_equation_string_y += " + x[" + str(Cu_sum_idx + 3*len(x_O)) + "] *" + str(  polarizability_O*(1E-24)* ( Cu_to_O_vec[1] /(Cu_to_O_dist**3) )  )

        
        #Filling the O dipole equation string array entries
        equation_string_array[2*O_idx] += O_equation_string_x.replace(" ","")
        equation_string_array[2*O_idx+1] += O_equation_string_y.replace(" ","")
        
        
            
    #### Clusters Equations
    for cluster_idx, cluster in enumerate(x_cluster):
        Cu_pos = cluster[0]
        for O_idx, O_pos in enumerate(cluster[1:]): 
            
            
            ### O (dipoles, charges) --> Clusters
            for O_sum_idx, O_sum_pos in enumerate(x_O): 
                
                O_to_O_vec = n.subtract(O_pos, O_sum_pos) * a
                O_to_O_dist = n.linalg.norm(O_to_O_vec)
                
                O_to_Cu_vec = n.subtract(Cu_pos, O_sum_pos) * a
                O_to_Cu_dist = n.linalg.norm(O_to_Cu_vec)
                
                
                ### O dipoles --> Clusters
                if O_to_O_dist > 0:
                    delta_clusters[cluster_idx][O_idx] +=  " + ec * ( (x[" + str(2*O_sum_idx) + "]*" + str(O_to_O_vec[0]) + "+ x[" + str(2*O_sum_idx+1) + "]*" + str(O_to_O_vec[1]) + ")/" + str(O_to_O_dist**3) + ")" 
                                                                                  
                delta_clusters[cluster_idx][O_idx] +=  " + ec * ( (x[" + str(2*O_sum_idx) + "]*" + str(-O_to_Cu_vec[0]) + "+ x[" + str(2*O_sum_idx+1) + "]*" + str(-O_to_Cu_vec[1]) + ")/" + str(O_to_Cu_dist**3) + ")"
                
                
                ### O charges --> Clusters
                if O_to_O_dist > 0:
                    delta_clusters[cluster_idx][O_idx] +=  "+ ec * x[" + str(O_sum_idx + 2*len(x_O)) + "] *" + str(   1/O_to_O_dist   )
                else:
                    delta_clusters[cluster_idx][O_idx] += " + EA_IP_func(x[" + str(O_sum_idx + 2*len(x_O)) + "]," + str(O_pos) + "," + str(hole_positions_array) +  ")"

                delta_clusters[cluster_idx][O_idx] +=  "+ ec * x[" + str(O_sum_idx + 2*len(x_O)) + "] *" + str(   -1/O_to_Cu_dist   )

           
            
            ### Cu charges --> Clusters
            for Cu_sum_idx, Cu_sum_pos in enumerate(x_Cu): 
                
                 Cu_to_Cu_vec = n.subtract(Cu_pos, Cu_sum_pos) * a
                 Cu_to_Cu_dist = n.linalg.norm(Cu_to_Cu_vec)
                 
                 Cu_to_O_vec = n.subtract(O_pos, Cu_sum_pos) * a
                 Cu_to_O_dist = n.linalg.norm(Cu_to_O_vec)
                
                 ### Cu charges --> Clusters
                 if Cu_to_Cu_dist > 0:
                     delta_clusters[cluster_idx][O_idx] +=  "+ ec * x[" + str(Cu_sum_idx + 3*len(x_O)) + "] *" + str(   -1/Cu_to_Cu_dist   )
                 else:
                     delta_clusters[cluster_idx][O_idx] += " + EA_IP_func(x[" + str(Cu_sum_idx + 3*len(x_O)) + "]," + str(Cu_pos) + "," + str(hole_positions_array) + ")"
                                                               
                 delta_clusters[cluster_idx][O_idx] +=  "+ ec * x[" + str(Cu_sum_idx + 3*len(x_O)) + "] *" + str(   1/Cu_to_O_dist   )
    


        #Filling the Cu charge equation string array entries
        equation_string_array[cluster_idx + 3*len(x_O)] += "+ec*H_bonds(" + str(tdp) + "," + str(tpp) + "," + delta_clusters[cluster_idx][0] + "," +  delta_clusters[cluster_idx][1]+ "," + delta_clusters[cluster_idx][2]+ "," + delta_clusters[cluster_idx][3] + ")[0]" 

    #Filling the O charge equation string array entries
    for O_idx, O_pos in enumerate(x_O):
         
        #Determining which clusters contains the O
        clusters = [cluster for cluster in x_cluster if O_pos in cluster ]
        cluster_indices = [x_cluster.index(cluster) for cluster in clusters]
        bond_indices = [cluster.index(O_pos) for cluster in clusters]
        for (cluster_idx, bond_idx) in zip(cluster_indices,bond_indices):
            equation_string_array[O_idx + 2*len(x_O)] += "+ec*H_bonds(" + str(tdp) + "," + str(tpp) + "," + delta_clusters[cluster_idx][0] + "," +  delta_clusters[cluster_idx][1]+ "," + delta_clusters[cluster_idx][2] + "," + delta_clusters[cluster_idx][3] + ")[" + str(bond_idx) + "]"

    
    #### Solving System of Equations
    def equations_func(x):
        equations_list = n.zeros(len(equation_string_array)) 
        for i in range(len(equation_string_array)):
            equations_list[i] = eval(equation_string_array[i])   
        return equations_list

    
    #Scaling all the dipole variables in each equation to get higher numbers
    x_scale_array = [10**(math.floor(math.log(n.abs(initial_guess[i]), 10))) if initial_guess[i] != 0 else  1E-20  for i in range(len(initial_guess))]
    for equation_index in range(len(equation_string_array)) :
        for dipole_index in range(len(equation_string_array)):
            equation_string_array[equation_index] = equation_string_array[equation_index].replace("x["+str(dipole_index)+"]", str(x_scale_array[dipole_index]) + "*x["+str(dipole_index)+"]")
    
    scaled_initial_guess = [i/x_scale_array[idx] for idx,i in enumerate(initial_guess)]
    results_verbose = optimize.fsolve(equations_func, scaled_initial_guess, epsfcn = 1E-6, xtol = 1E-5,  full_output=True)
    results = [x_scale_array[idx] * i for idx,i in enumerate(results_verbose[0])] 
    
    
    #Calculating the Charge Transfer Value for all bonds after convergence (IN eV, NOT erg)
    def delta_array_eval(x):
        delta_evaluated = n.zeros((len(delta_clusters),4))
        for cluster_idx,_ in enumerate(delta_clusters):
            for bond_idx,_ in enumerate(delta_clusters[1]):
                delta_evaluated[cluster_idx][bond_idx] = eval(delta_clusters[cluster_idx][bond_idx])
            
        return delta_evaluated/ev_to_erg
    
    delta_clusters_converged = delta_array_eval(results)
    
   
    #Array of all converged and initial charges with structure [[q1, position1], ... ]   
    charges = [[q,pos] for (q,pos) in zip(results[2*len(x_O):3*len(x_O)], x_O)] + [[q,pos] for (q,pos) in zip(results[3*len(x_O):], x_Cu)]     
    charges_with_holes = [ [q + ec * hole_positions_array.count(pos), pos] for (q,pos) in charges]
    charges_initial = [[q,pos] for (q,pos) in zip(initial_guess_O, x_O)] + [[q,pos] for (q,pos) in zip(initial_guess_Cu, x_Cu)] 
    charges_with_holes_initial = [ [q + ec * hole_positions_array.count(pos), pos] for (q,pos) in charges_initial]
    

    ####Monopole Energy
    energy_charges, energy_charges_initial = 0,0
    energy_VM, energy_VM_initial = 0,0
    energy_U, energy_U_initial = 0,0
    for _, ( (q,pos),(q_init,_)) in enumerate(zip(charges_with_holes, charges_with_holes_initial)):
        
        
        
        ## Interaction between charges (different sites and includes the doped holes)
        for _, ( (q_other,pos_other),(q_init_other,_)) in enumerate(zip(charges_with_holes, charges_with_holes_initial)):
            Rij = n.linalg.norm(n.subtract(pos, pos_other))*a
            if Rij > 0:
                energy_charges += (q*q_other)/(2*Rij)
                energy_charges_initial += (q_init*q_init_other)/(2*Rij)

        
        
        ## EA/IP cost of putting the charges on their sites and Madelung Energy
        #Oxygen sites
        if int(pos[0] + pos[1]) != (pos[0] + pos[1]):
            
            ## Bare Madelung Potential to charges
            energy_VM += VM_bare * q/ec
            energy_VM_initial += VM_bare * q_init/ec
            
            if  0 <= q/ec <= 1:
                energy_U += -EAO_2*q/ec
            elif 1 < q/ec <= 2:
                energy_U += -EAO_2 - EAO_1 * (q/ec - 1)
            elif 2 < q/ec <= 3:
                energy_U += -EAO_2 - EAO_1 + IPO_1 * (q/ec - 2)

                
            if  0 <= q_init/ec <= 1:
                energy_U_initial += -EAO_2*q_init/ec
            elif 1 < q_init/ec <= 2:
                energy_U_initial += -EAO_2 - EAO_1 * (q_init/ec - 1)
            elif 2 < q_init/ec <= 3:
                energy_U_initial += -EAO_2 - EAO_1 + IPO_1 * (q_init/ec - 2)
        
            
        #Cu sites
        else:
            
            ## Bare Madelung Potential to charges
            energy_VM += -VM_bare * q/ec
            energy_VM_initial += -VM_bare * q_init/ec
            
            if  0 <= q/ec <= 1:
                energy_U += IPCU_2*q/ec
            elif 1 < q/ec <= 2:
                energy_U += IPCU_2 + IPCU_3 * (q/ec - 1)
            elif 2 < q/ec <= 3:
                energy_U += IPCU_2 + IPCU_3 + IPCU_4 * (q/ec - 2)

                
            if  0 <= q_init/ec <= 1:
                energy_U_initial += IPCU_2*q_init/ec
            elif 1 < q_init/ec <= 2:
                energy_U_initial += IPCU_2 + IPCU_3 * (q_init/ec - 1)
            elif 2 < q_init/ec <= 3:
                energy_U_initial += IPCU_2 + IPCU_3 + IPCU_4 * (q_init/ec - 2)

        
     
  
    ####Dipole Energy
    monopole_field_at_dipoles = n.zeros((len(x_O),2))
    for dip_idx,dip_pos in enumerate(x_O):
        
        #Charges to Dipoles
        for (q_mono, pos_mono) in charges_with_holes:
            mono_to_dip_vec = n.subtract(dip_pos,pos_mono)*a
            mono_to_dip_dist = n.linalg.norm(mono_to_dip_vec)
            if mono_to_dip_dist > 0:
                monopole_field_at_dipoles[dip_idx] += n.array(q_mono * mono_to_dip_vec/(mono_to_dip_dist**3) )
        
    O_dipoles = n.array([[results[2*O_index], results[2*O_index+1]] for O_index,_ in enumerate(x_O)])
    energy_dipoles = -0.5*n.einsum("ij,ij", monopole_field_at_dipoles, O_dipoles)
    energy_dipoles_initial = -0.5*n.einsum("ij,ij",initial_monopole_field_at_dipoles, initial_guess_O_dipoles)
    
    ###    
    
    #Adding up charge and dipole contributions to the total interaction energy
    energy = (energy_charges + energy_dipoles + energy_VM + energy_U)
    energy_initial = (energy_charges_initial + energy_dipoles_initial + energy_VM_initial + energy_U_initial)
    
    #Priting energies, equation convergence data and time taken
    print("!Second Hole Position = ", second_hole_pos)
    print("-> Total Energy =", energy)
    print("Energy (Charges) =", (energy_charges)/ev_to_erg)
    print("Energy (Dipoles) =", energy_dipoles/ev_to_erg)
    print("-> Total Energy Initial =", energy_initial)
    print("Energy initial (Charges) =", (energy_charges_initial)/ev_to_erg)
    print("Energy initial (Dipoles) =", energy_dipoles_initial/ev_to_erg)
    print("--- %s,  Iterations: %d ---" %(results_verbose[3], results_verbose[1]['nfev'] ))
    print("--- %s minutes ---\n" % ((time.time() - start_time)/60))
    
    
    #Creating a dictionary to return all the quantities I need
    results_dict = {}
    results_dict.update({"results": results, "initial guess": initial_guess, "x_O": x_O, "x_Cu": x_Cu, "x_cluster": x_cluster, "hole positions array": hole_positions_array, "second hole pos": second_hole_pos, "delta": delta_clusters_converged, "delta initial": delta_clusters_initial/ev_to_erg } )
    results_dict.update({"charges": charges, "charges with holes": charges_with_holes, "charges initial": charges_initial, "O dipoles": O_dipoles, "O dipoles initial": initial_guess_O_dipoles})
    results_dict.update({"energy": energy/ev_to_erg, "energy charges": energy_charges/ev_to_erg, "energy dipoles": energy_dipoles/ev_to_erg, "energy Madelung": energy_VM/ev_to_erg, "energy U": energy_U/ev_to_erg,  "energy initial": energy_initial/ev_to_erg, "energy charges initial": energy_charges_initial/ev_to_erg, "energy Madelung initial": energy_VM_initial/ev_to_erg, "energy U initial": energy_U_initial/ev_to_erg, "energy dipoles initial": energy_dipoles_initial/ev_to_erg})
    results_dict.update({"results verbose": results_verbose})

    return results_dict


#%% Exporting Results and Multhreading

#Importing Parameters
lattice_constant = 3.85E-8
r_0 = int(sys.argv[1])
max_dist = int(sys.argv[2])
tdp = float(sys.argv[3])
tpp = float(sys.argv[4])
delta_0 = float(sys.argv[5])
pol_O = float(sys.argv[6])
first_hole_atom = int(sys.argv[7])
second_hole_atom = int(sys.argv[8])
hole_config = int(sys.argv[9])

#Making some strings based on the type of atoms the holes are placed in and which holes I'm considered at all
hole_config_string = "No_Holes" if hole_config == 0 else "Two_Holes" if hole_config == 1 else "First_Hole" if hole_config == 2 else "Second_Hole"
first_hole_atom_string = "Cu" if first_hole_atom == 0 else "O" 
second_hole_atom_string = "Cu" if second_hole_atom == 0 else "O" 



#Determining the fixed position of the first hole and the variable positions of the second hole
first_hole_pos = [0,0] if first_hole_atom == 0 else [0.5,0]
trial_positions = []
for x in range(0,max_dist+1):
    for y in range(0,max_dist+1):
        
            #Cu sites
            if second_hole_atom == 0:
                trial_positions.append([x, y])
            
            #O sites
            elif second_hole_atom == 1:
                trial_positions.append([x + 0.5, y])
                trial_positions.append([x - 0.5, y])
                trial_positions.append([x , y + 0.5])
                trial_positions.append([x, y - 0.5])
      
#Function to sort positoins by norm from first_hole_pos
def norm_sorting(second_hole_pos):
    return n.linalg.norm(n.subtract(second_hole_pos, first_hole_pos))

#Making separated lists of the the two different regions (up region, right region)
second_hole_pos = sorted(n.unique([pos for pos in trial_positions if pos[0] >= first_hole_pos[0] and pos[1] >= first_hole_pos[1] ], axis = 0).tolist(), key = norm_sorting)   
second_hole_pos_up = [pos for pos in second_hole_pos if pos[1]-first_hole_pos[1] >= pos[0]-first_hole_pos[0]]
second_hole_pos_right = [pos for pos in second_hole_pos if pos[0]-first_hole_pos[0] >= pos[1]-first_hole_pos[1]]
    

#Printing the Paramters for this calculation
print("%s exact 5x5 Calculations: r0 = %d, Max hole-hole distance: %d, tdp = %.2f, tpp = %.2f, delta = %.2f, pol_O = %.2f, First Hole on %s [%.1f, %.1f], Second Hole on %s " % (hole_config_string.replace("_", " ") ,r_0, max_dist, tdp, tpp, delta_0, pol_O, first_hole_atom_string, first_hole_pos[0], first_hole_pos[1], second_hole_atom_string))

#Multithreading
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

arguments = [(first_hole_pos, i, lattice_constant, pol_O, tdp, tpp, delta_0, r_0, hole_config) for i in second_hole_pos]

with Pool(number_of_cores) as pool:
	all_results = pool.starmap(cluster_energy_CuO, arguments)

#Exporting the results in a dictionary with structure, with the second hole position as the label for each results
all_results_dict = {"second hole positions up": second_hole_pos_up, "second hole positions right": second_hole_pos_right}
for results in all_results:
    all_results_dict.update({str(results["second hole pos"]):results})

with open("r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d_more_pos.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, hole_config_string, max_dist), 'wb') as f:
    pickle.dump(all_results_dict, f)


print("\n--- TOTAL TIME: %s minutes ---" % ((time.time() - total_time)/60))




