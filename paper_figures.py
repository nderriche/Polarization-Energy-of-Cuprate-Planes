import numpy as n
import pickle
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as mpe
import matplotlib.patches as patches
import scipy.linalg as la
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D

#Constants
ec = 4.8E-10
lattice_constant = 3.85E-8
delta_0_base = 6.0
tdp_base = 1.3
tpp_base = 0.65
pol_O_base = 2.75
c_base = 1
bare_Upp = 9.171
bare_Udd = 20.539
ev_to_erg = 1.6021773E-12
EAO_1 = -1.461 
EAO_2 = 7.72  


# Parameter value lists
delta_array_converged = [6.0, 4.5, 9.0, 12.0]
tdp_array_converged = [1.3, 0.97, 1.95, 2.6, 5.2]
tpp_array_converged = [0.65, 0.0, 0.325, 0.4875, 0.975]
pol_array_converged = [2.75, 1.375, 2.0625, 3.4375]






#%% Figure 2: V'(R) and Contributions with Base Parameters

### Importing Polarization Energy Data
r_0 = 4
max_dist = 8
xlim = max_dist
type_of_energy = "energy"
first_hole_atom_string = "O"
second_hole_atom_string = "O"
delta_0 = delta_0_base
tdp = tdp_base
tpp = tpp_base
pol_O = pol_O_base
subtracted_type_of_energy = "energy Madelung"
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Two_Holes", max_dist), 'rb') as f:
    data_two_holes = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "First_Hole", max_dist), 'rb') as f:
    data_first_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Second_Hole", max_dist), 'rb') as f:
    data_second_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "No_Holes", max_dist), 'rb') as f:
    data_no_holes = pickle.load(f)
    
    
    
### Figure Configuration
multiplier_res = 6
dpi=600
#fig = plt.figure(figsize=(1.3*multiplier_res, 1*multiplier_res), dpi=dpi)
#spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
#ax = [fig.add_subplot(spec[0,0]), fig.add_subplot(spec[1,0]), fig.add_subplot(spec[0,1]), fig.add_subplot(spec[1,1])]
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(1.3*multiplier_res, 1*multiplier_res), dpi=dpi)
ax = ax.flatten(order="F")
x_lim = 4
y_lim = 4
plot_lim = 4
label_size = 16
tick_size = 13
legend_size = 12
line_width = 3
arrowhead_width = 0.35
color_map = plt.cm.BrBG
color_list = ['#1f77b4', "firebrick", "chocolate", "green", "magenta" ]
marker_list = ["o", "^", "s", "*", "D"]
legend_labels = ["$V'$", "Atomic Dipoles", "Bond Dipoles"]
energy_types = ["energy", "energy initial", "energy charges", "energy dipoles", "energy U"]
energy_types = ["energy", "energy initial", "energy dipoles", "energy charges", "energy U"]

    
#Hole positions and distance arrays
first_hole_pos = [0,0] if first_hole_atom_string == "Cu" else [0.5,0]
second_hole_pos_up = data_two_holes["second hole positions up"]
second_hole_pos_right = data_two_holes["second hole positions right"]
dist_up = [n.linalg.norm(n.subtract(i,first_hole_pos)) for i in second_hole_pos_up]
dist_right = [n.linalg.norm(n.subtract(i,first_hole_pos)) for i in second_hole_pos_right]
dist_antinodal = [i[0]-0.5 for i in second_hole_pos_right if i[1] == 0.0]
dist_nodal = [i[1] for i in second_hole_pos_up if i[0]-0.5 == 0.0]
energy_bond_dipoles_antinodal = n.zeros(len(dist_antinodal))
energy_bond_dipoles_nodal = n.zeros(len(dist_nodal))
for idx, type_of_energy in enumerate(energy_types):
    #Energy of different hole configurations, depending on if I'm subtracting a contribution to the energy
    energy_two_holes_up = n.array([ data_two_holes[str(i)][type_of_energy] - data_two_holes[str(i)][subtracted_type_of_energy]  for i in second_hole_pos_up ]) 
    energy_two_holes_right = n.array([ data_two_holes[str(i)][type_of_energy] - data_two_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
    energy_first_hole_up = n.array([ data_first_hole[str(i)][type_of_energy] - data_first_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
    energy_first_hole_right = n.array([ data_first_hole[str(i)][type_of_energy] - data_first_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
    energy_second_hole_up = n.array([ data_second_hole[str(i)][type_of_energy] - data_second_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
    energy_second_hole_right = n.array([ data_second_hole[str(i)][type_of_energy] - data_second_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
    energy_no_holes_up = n.array([ data_no_holes[str(i)][type_of_energy] - data_no_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
    energy_no_holes_right = n.array([ data_no_holes[str(i)][type_of_energy] - data_no_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right]) 

    energy_subtracted_up = n.array([energy_two_holes_up[i] - energy_first_hole_up[i] - energy_second_hole_up[i] + energy_no_holes_up[i] for i,_ in enumerate(energy_two_holes_up)])
    energy_subtracted_right = n.array([energy_two_holes_right[i] - energy_first_hole_right[i] - energy_second_hole_right[i] + energy_no_holes_right[i] for i,_ in enumerate(energy_two_holes_right)])
    energy_subtracted_antinodal = [i for idx,i in enumerate(energy_subtracted_right) if second_hole_pos_right[idx][1] == 0.0]
    energy_subtracted_nodal = [i for idx,i in enumerate(energy_subtracted_up) if second_hole_pos_up[idx][0]-0.5 == 0.0]


    ### b) Comparison with V0
    if type_of_energy == "energy":
        #Bare Potential
        bare_U = bare_Udd
        dist_V0 = n.linspace(0.1, xlim, 100)
        V0 = [(ec**2)/(abs(i)*lattice_constant)*6.242e+11 for i in dist_V0 ]
        ax[2].plot(dist_V0, V0, color = 'black', linewidth = line_width, label = r"$V_0$", zorder = -1)
        
        #Antiodal Direction
        ax[2].plot(dist_antinodal, energy_subtracted_antinodal, linewidth=line_width, color="fuchsia", marker = "o", label=r"$V'$ (O-Cu)")
        ax[1].plot(dist_antinodal, energy_subtracted_antinodal, linewidth=line_width, color="fuchsia", marker = "o", label=r"$V'$ (O-Cu)")
        #Nodal Direction
        ax[2].plot(dist_nodal, energy_subtracted_nodal, linewidth=line_width, color="dodgerblue", marker = "o", label=r"$V'$ (O-O)")
        ax[3].plot(dist_nodal, energy_subtracted_nodal, linewidth=line_width, color="dodgerblue", marker = "o", label=r"$V'$ (O-O)")
        
    elif type_of_energy == "energy initial":
        #Antiodal Direction
        ax[2].plot(dist_antinodal, energy_subtracted_antinodal, linewidth=line_width, color="fuchsia", linestyle="dashed", label=r"$V'$ (O-Cu, no LFE)")
        #Nodal Direction
        ax[2].plot(dist_nodal, energy_subtracted_nodal , linewidth=line_width, color="dodgerblue", linestyle="dashed", label=r"$V'$ (O-O, no LFE)")
        
    
    
    ### c)-d) Different Terms   
    elif type_of_energy == "energy dipoles":
        #Antiodal Direction
        ax[1].plot(dist_antinodal, energy_subtracted_antinodal, linewidth=line_width, color=color_list[idx-1], marker = marker_list[idx-1], label="Atomic Dipoles")
        #Nodal Direction
        ax[3].plot(dist_nodal, energy_subtracted_nodal, linewidth=line_width, color=color_list[idx-1], marker = marker_list[idx-1], label="Atomic Dipoles")


    else:
        energy_bond_dipoles_antinodal = energy_bond_dipoles_antinodal + n.array(energy_subtracted_antinodal)
        energy_bond_dipoles_nodal = energy_bond_dipoles_nodal + n.array(energy_subtracted_nodal)
    
#Antinodal direction
ax[1].plot(dist_antinodal, energy_bond_dipoles_antinodal, linewidth=line_width, color=color_list[idx-1], marker = marker_list[idx-1], label="Bond Dipoles")
#Nodal Direction
ax[3].plot(dist_nodal, energy_bond_dipoles_nodal, linewidth=line_width, color=color_list[idx-1], marker = marker_list[idx-1], label="Bond Dipoles")

        
        
        
        
        
        
### a) Real Space Color Plot
r_0 = 4
max_dist = 6
xlim = max_dist
type_of_energy = "energy"
first_hole_atom_string = "O"
second_hole_atom_string = "O"
delta_0 = delta_0_base
tdp = tdp_base
tpp = tpp_base
pol_O = pol_O_base
subtracted_type_of_energy = "energy Madelung"
energy_types = ["energy", "energy charges", "energy dipoles", "energy U"][0:1]
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d_more_pos.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Two_Holes", max_dist), 'rb') as f:
    data_two_holes = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d_more_pos.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "First_Hole", max_dist), 'rb') as f:
    data_first_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d_more_pos.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Second_Hole", max_dist), 'rb') as f:
    data_second_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d_more_pos.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "No_Holes", max_dist), 'rb') as f:
    data_no_holes = pickle.load(f)
    
#Hole positions and distance arrays
first_hole_pos = [0,0] if first_hole_atom_string == "Cu" else [0.5,0]
second_hole_pos_up = data_two_holes["second hole positions up"]
second_hole_pos_right = data_two_holes["second hole positions right"]
dist_up = [n.linalg.norm(n.subtract(i,first_hole_pos)) for i in second_hole_pos_up]
dist_right = [n.linalg.norm(n.subtract(i,first_hole_pos)) for i in second_hole_pos_right]
for idx, type_of_energy in enumerate(energy_types):
    #Energy of different hole configurations, depending on if I'm subtracting a contribution to the energy
    energy_two_holes_up = n.array([ data_two_holes[str(i)][type_of_energy] - data_two_holes[str(i)][subtracted_type_of_energy]  for i in second_hole_pos_up ]) 
    energy_two_holes_right = n.array([ data_two_holes[str(i)][type_of_energy] - data_two_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
    energy_first_hole_up = n.array([ data_first_hole[str(i)][type_of_energy] - data_first_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
    energy_first_hole_right = n.array([ data_first_hole[str(i)][type_of_energy] - data_first_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
    energy_second_hole_up = n.array([ data_second_hole[str(i)][type_of_energy] - data_second_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
    energy_second_hole_right = n.array([ data_second_hole[str(i)][type_of_energy] - data_second_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
    energy_no_holes_up = n.array([ data_no_holes[str(i)][type_of_energy] - data_no_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
    energy_no_holes_right = n.array([ data_no_holes[str(i)][type_of_energy] - data_no_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right]) 

    energy_subtracted_up = n.array([energy_two_holes_up[i] - energy_first_hole_up[i] - energy_second_hole_up[i] + energy_no_holes_up[i] for i,_ in enumerate(energy_two_holes_up)])
    energy_subtracted_right = n.array([energy_two_holes_right[i] - energy_first_hole_right[i] - energy_second_hole_right[i] + energy_no_holes_right[i] for i,_ in enumerate(energy_two_holes_right)])


#Create array with real-space screened potential
potential_real_data = []
for pos_up, energy_up, pos_right, energy_right in zip(second_hole_pos_up, energy_subtracted_up.tolist(), second_hole_pos_right, energy_subtracted_right.tolist()):
    potential_real_data.append((pos_up[0]-0.5, pos_up[1], energy_up))
    potential_real_data.append((pos_right[0]-0.5, pos_right[1], energy_right))
potential_real_data = sorted(list(set(potential_real_data)), key=lambda i: n.linalg.norm(i[0:2]))
x, y, z = zip(*potential_real_data)

#Plotting the four quadrants
num_levels = 20
contour_ax_upright = ax[0].tricontourf(x,y,z, cmap=color_map, levels=num_levels)
contour_ax_lowerright = ax[0].tricontourf(x,[-i for i in y],z, cmap=color_map, levels=num_levels)
contour_ax_upleft = ax[0].tricontourf([-i for i in x],y,z, cmap=color_map, levels=num_levels)
contour_ax_lowerleft = ax[0].tricontourf([-i for i in x],[-i for i in y],z, cmap=color_map, levels=num_levels)
cbar = fig.colorbar(contour_ax_upright, location="right", pad=0.05)

#Adding dashed lines to represent the 1D plots
outline=mpe.withStroke(linewidth=2.5, foreground='black')
ax[0].axhline(0, xmin=0.5, linewidth=line_width/2, linestyle="dashed", color="fuchsia", path_effects=[outline])
ax[0].arrow(3.45,0, dx=0.05, dy=0, head_width=arrowhead_width, zorder=10, edgecolor="black", facecolor="fuchsia")
ax[0].axvline(0, ymin=0.5, linewidth=line_width/2, linestyle="dashed", color="dodgerblue", path_effects=[outline])
ax[0].arrow(0,3.4, dx=0, dy=0.05, head_width=arrowhead_width, zorder=10, edgecolor="black", facecolor="dodgerblue")

    


### Plotting Parameters
#ax[0].set_aspect("equal")
ax[0].set_xlim(-plot_lim,plot_lim)
ax[0].set_ylim(-plot_lim,plot_lim)
ax[0].set_xticks(n.arange(-plot_lim, plot_lim+0.5, 2))
ax[0].set_yticks(n.arange(-plot_lim, plot_lim+0.5, 2))
ax[0].set_xlabel("x (R/a)", fontsize=label_size)
ax[0].set_ylabel("y (R/a)", fontsize=label_size, labelpad=1)
ax[0].tick_params(axis="both", labelsize=tick_size)
#ax[0].yaxis.set_label_position("right")
#ax[0].yaxis.tick_right()
ax[0].set_anchor("NW")
cbar.set_label("Energy (eV)", fontsize=label_size, labelpad=4.5)
cbar.ax.tick_params(labelsize=tick_size)
for idx in range(1,4):
    ax[idx].set_anchor("W")
    ax[idx].set_xlabel("R/a", fontsize=label_size)
    ax[idx].set_ylabel("Energy (eV)", fontsize=label_size, labelpad=1)
    ax[idx].tick_params(axis="both", labelsize=tick_size)
    ax[idx].set_xlim(0,8)
    ax[idx].grid()
    if idx == 2:
        ax[idx].legend(loc="upper right", fontsize=legend_size/1.3)
        ax[idx].set_ylim(0,2)
        ax[idx].set_yticks([0,0.5,1,1.5,2.0]) 
    else:
        ax[idx].set_yticks([-6, -4, -2, 0, 2])
        ax[idx].legend(loc="lower right", fontsize=legend_size)
        
fig.tight_layout()








#%% Figure 3: Real Space Polarization Fields (Same Site, nn and 2nn)

#### Figure Creation Settings
fig_width = 3.375
fig_height = fig_width * 3/2
dpi = 800
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(fig_width, fig_height), dpi=dpi)
fig.subplots_adjust(wspace = 0.1 , hspace=0.1)
    
#Global sizes of plot elements
xlim = 3
ylim = 3
axis_label_fontsize = 17
tick_label_fontsize = 10
legend_fontsize = 5
line_width = 1.5
line_markersize = 2.2
charge_fontsize = 7
quiver_scale_interference = 2e-19
quiver_scale_separate = 1e-17

#Adding x-labels for the two categories
ax[2,0].set_xlabel(r"$[h_1 + h_2]$", labelpad = 13, weight='bold', fontsize = axis_label_fontsize)
ax[2,1].set_xlabel(r"$[h_1] + [h_2]$", labelpad = 13, weight='bold', fontsize = axis_label_fontsize)

#### Import Parameters
r_0 = 4
max_dist = 4
type_of_energy = "energy"
first_hole_atom_string = "O"
second_hole_atom_string = "O"
first_hole_pos = [0,0] if first_hole_atom_string == "Cu" else [0.5,0.0]

#Base Parameters
delta_0 = delta_0_base
tdp = tdp_base
tpp = tpp_base
pol_O = pol_O_base


# Making an iteratable dictionary with parameters for all Sub-Figs
subfigs_dict = {"a)": {}, "b)": {}, "c)": {}, "d)": {}, "e)": {}, "f)": {}}


subfigs_dict["a)"].update({"second hole position": [1.5,0.0], "configuration": "subtracted", "quiver scale": quiver_scale_interference, "subplot indices": [0,0]})
subfigs_dict["b)"].update({"second hole position": [1.5,0.0], "configuration": "individual holes subtracted", "quiver scale": quiver_scale_separate, "subplot indices": [0,1]})
subfigs_dict["c)"].update({"second hole position": [0.5,1.0], "configuration": "subtracted", "quiver scale": quiver_scale_interference,"subplot indices": [1,0]})
subfigs_dict["d)"].update({"second hole position": [0.5,1.0], "configuration": "individual holes subtracted","quiver scale": quiver_scale_separate, "subplot indices": [1,1]})
subfigs_dict["e)"].update({"second hole position": [1.0,0.5], "configuration": "subtracted", "quiver scale": quiver_scale_interference,"subplot indices": [2,0]})
subfigs_dict["f)"].update({"second hole position": [1.0,0.5], "configuration": "individual holes subtracted", "quiver scale": quiver_scale_separate, "subplot indices": [2,1]})



#Importing data files
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Two_Holes", max_dist), 'rb') as f:
    data_two_holes = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "First_Hole", max_dist), 'rb') as f:
    data_first_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Second_Hole", max_dist), 'rb') as f:
    data_second_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "No_Holes", max_dist), 'rb') as f:
    data_no_holes = pickle.load(f)

  

# Looping to Plot Subfigs
for subfig_label, subfig_params in subfigs_dict.items():
    
    '''
    #Skipping subfigs for time 
    if subfig_label not in ["a)"]:
        continue
    '''

    #### Plotting Settings
    subplot_idx = subfig_params["subplot indices"]
    second_hole_pos = subfig_params["second hole position"]
    hole_positions = [[0.5,0.0], second_hole_pos]
    
    #Plotting limits, and making sure the x range and y range are equal
    max_dist_from_hole = 1
    xlims = [min([hole_pos[0] for hole_pos in hole_positions]) - max_dist_from_hole, max([hole_pos[0] for hole_pos in hole_positions]) + max_dist_from_hole]
    ylims = [min([hole_pos[1] for hole_pos in hole_positions]) - max_dist_from_hole, max([hole_pos[1] for hole_pos in hole_positions]) + max_dist_from_hole]
    xrange = xlims[1] - xlims[0]
    yrange = ylims[1] - ylims[0]
    if yrange > xrange:
        xlims[0] = xlims[0] + (xrange-yrange)/2
        xlims[1] = xlims[1] - (xrange-yrange)/2
    else:
        ylims[0] = ylims[0] + (yrange-xrange)/2
        ylims[1] = ylims[1] - (yrange-xrange)/2
        
    lim_offset = 0.21
    ax[subplot_idx[0], subplot_idx[1]].set_xlim(xlims[0] + lim_offset, xlims[1] - lim_offset)
    ax[subplot_idx[0], subplot_idx[1]].set_ylim(ylims[0] + lim_offset, ylims[1] - lim_offset)
    
    #Adding background colours to represent Paths 1 and 2 (done by plotting three points as triangles, and dashed lines to separate regions)
    Path_1_Triangle_up = n.array([ hole_positions[0], n.add(hole_positions[0], [2,2]), n.add(hole_positions[0], [-2,2]), hole_positions[0]])
    Path_1_Triangle_down = n.array([ hole_positions[0], n.add(hole_positions[0], [2,-2]), n.add(hole_positions[0], [-2,-2]), hole_positions[0]])
    Path_2_Triangle_right = n.array([ hole_positions[0], n.add(hole_positions[0], [2,2]), n.add(hole_positions[0], [2,-2]), hole_positions[0]])
    Path_2_Triangle_left = n.array([ hole_positions[0], n.add(hole_positions[0], [-2,2]), n.add(hole_positions[0], [-2,-2]), hole_positions[0]])
    paths_triangles = n.array([Path_1_Triangle_up, Path_1_Triangle_down, Path_2_Triangle_right, Path_2_Triangle_left])
    triangle_color_array = ["#85ffffff", "#85ffffff", "#dbc5ffff", "#dbc5ffff"]
    
    for (triangle_idx, triangle_points) in enumerate(paths_triangles):
        ax[subplot_idx[0], subplot_idx[1]].plot( [i[0] for i in triangle_points], [i[1] for i in triangle_points], linestyle = "",  zorder=-2) #Region 1 (up)
        ax[subplot_idx[0], subplot_idx[1]].fill( [i[0] for i in triangle_points], [i[1] for i in triangle_points], linestyle = "", facecolor = triangle_color_array[triangle_idx], alpha = 0.5, zorder=-2) #Region 1 (up)
     
    #Removing axis ticks
    ax[subplot_idx[0], subplot_idx[1]].set_xticklabels([])
    ax[subplot_idx[0], subplot_idx[1]].set_yticklabels([])
    ax[subplot_idx[0], subplot_idx[1]].tick_params(axis='both', which='both',bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
    

    #Configuration type
    configuration = subfig_params["configuration"]
    if configuration == "two holes":
        energy_results = data_two_holes
        dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
        mono_charges = energy_results[str(second_hole_pos)]["charges"]
        dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )
    
    elif configuration == "first hole":
        hole_positions = hole_positions[0:1]
        energy_results = data_first_hole
        dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
        mono_charges = energy_results[str(second_hole_pos)]["charges"]
        dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )
    
    elif configuration == "second hole":
        hole_positions = hole_positions[1:2]
        energy_results = data_second_hole
        dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
        mono_charges = energy_results[str(second_hole_pos)]["charges"]
        dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )
    
    elif configuration == "no holes":
        hole_positions = []
        energy_results = data_no_holes
        dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
        mono_charges = energy_results[str(second_hole_pos)]["charges"]
        dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )
    
    
    elif configuration == "sum of holes 1 and 2":
        energy_results = data_second_hole
        mono_charges = [ [(q1[0] + q2[0])/2, q1[1]] if q1[1] not in hole_positions else [0, q1[1]] for (q1,q2) in zip(data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"])]
        dipoles = n.add( data_first_hole[str(second_hole_pos)]["O dipoles"], data_second_hole[str(second_hole_pos)]["O dipoles"] )/2
        dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])
        
    
    elif configuration == "subtracted":
        energy_results = data_two_holes
        mono_charges = [ [q_two_holes[0] - q_first_hole[0] - q_second_hole[0] + q_no_holes[0], q_two_holes[1]] for (q_two_holes,q_first_hole,q_second_hole,q_no_holes) in zip(data_two_holes[str(second_hole_pos)]["charges"], data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"], data_no_holes[str(second_hole_pos)]["charges"])]
        dipoles = n.add( n.subtract( n.subtract( data_two_holes[str(second_hole_pos)]["O dipoles"], data_first_hole[str(second_hole_pos)]["O dipoles"] ), data_second_hole[str(second_hole_pos)]["O dipoles"]), data_no_holes[str(second_hole_pos)]["O dipoles"] )
        dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])
    
    elif configuration == "subtracted only no holes":
        energy_results = data_two_holes
        mono_charges = [ [q_two_holes[0] - q_no_holes[0], q_two_holes[1]] for (q_two_holes,q_first_hole,q_second_hole,q_no_holes) in zip(data_two_holes[str(second_hole_pos)]["charges"], data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"], data_no_holes[str(second_hole_pos)]["charges"])]
        dipoles = n.subtract( data_two_holes[str(second_hole_pos)]["O dipoles"], data_no_holes[str(second_hole_pos)]["O dipoles"] )
        dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])
    
    elif configuration == "individual holes subtracted":
        energy_results = data_first_hole
        mono_charges = [ [q_first_hole[0] + q_second_hole[0] - 2*q_no_holes[0], q_first_hole[1]] for (q_first_hole,q_second_hole,q_no_holes) in zip(data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"], data_no_holes[str(second_hole_pos)]["charges"])]
        dipoles = n.subtract( n.subtract( n.add( data_first_hole[str(second_hole_pos)]["O dipoles"], data_second_hole[str(second_hole_pos)]["O dipoles"] ), data_no_holes[str(second_hole_pos)]["O dipoles"]), data_no_holes[str(second_hole_pos)]["O dipoles"] )
        dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])
        
    #Imported Position Arrays
    x_O = n.array(energy_results[str(second_hole_pos)]["x_O"])
    x_O_dipoles = n.array([x for x in x_O if x.tolist() not in hole_positions])
    x_Cu = n.array(energy_results[str(second_hole_pos)]["x_Cu"])

    #### Plotting Coloured Dots
    #O and Cu Sites
    for i in range(len(x_O)):
        ax[subplot_idx[0], subplot_idx[1]].scatter(x_O[i][0], x_O[i][1], color='#dd1308ff')
    for i in range(len(x_Cu)):
        #ax[subplot_idx[0], subplot_idx[1]].scatter(x_Cu[i][0], x_Cu[i][1], color='#b8860bff', s=11, zorder=2)
        ax[subplot_idx[0], subplot_idx[1]].scatter(x_Cu[i][0], x_Cu[i][1], color='saddlebrown', s=11, zorder=2)
    
    #+1 Doped Holes
    for hole_pos in hole_positions:
        ax[subplot_idx[0], subplot_idx[1]].scatter(hole_pos[0], hole_pos[1], color='#05ca1cff', s = 100, marker = "x")
        ax[subplot_idx[0], subplot_idx[1]].annotate("+" + str(hole_positions.count(hole_pos)), xy=(hole_pos[0],hole_pos[1]), fontsize=5.5, color = "#ffff00ff", weight='bold', ha="center", va="center", path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    
    ###Plotting the O dipoles
    quiver_scale_O = subfig_params["quiver scale"]
    ax[subplot_idx[0], subplot_idx[1]].quiver(x_O_dipoles.T[0], x_O_dipoles.T[1], dipoles[0:-1:2], dipoles[1:len(dipoles):2], color = "#00ff00ff", edgecolor = "black", scale = quiver_scale_O, pivot="mid", label="O Dipoles", headlength = 5.5*1.4, headaxislength = 5.5, headwidth = 5.5, linewidth = 1, width = 0.02, zorder = 2)

    ###Writing the induced charges
    for q_idx, (q, bond_pos) in enumerate(mono_charges):
        if int(bond_pos[0] + bond_pos[1]) != (bond_pos[0] + bond_pos[1]):
            #color = "#ad0e98ff"
            color = "orange"
        else:
            #color = "#0b3db8ff"
            color = "dodgerblue"
            
        if xlims[0] < bond_pos[0] < xlims[1] and ylims[0] < bond_pos[1] < ylims[1]:
            ax[subplot_idx[0], subplot_idx[1]].text(bond_pos[0], bond_pos[1] + 0.28*0.6, "%d" % (q/ec*1000), fontsize = charge_fontsize,  color = color, weight="heavy", path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],  verticalalignment = "center", horizontalalignment = "center" , zorder = 6)


    #Subfig label text
    ax[subplot_idx[0], subplot_idx[1]].text(0.04, 0.85, subfig_label, fontsize = 2.5*legend_fontsize, transform = ax[subplot_idx[0], subplot_idx[1]].transAxes, weight = "bold", zorder = 10)



    # Total dipole electric field at holes
    O_dipoles = [n.array([i,j]) for (i,j) in zip(dipoles[0:-1:2], dipoles[1:len(dipoles):2])]
    for hole_index, hole_pos in enumerate(hole_positions):
        electric_field_charges = n.array([0.0,0.0])
        electric_field_dipoles = n.array([0.0,0.0])
        
        #Monopole electric field
        for q_idx, (q, q_pos) in enumerate(mono_charges):
            
            q_to_hole_vec = n.subtract(n.array(hole_pos), n.array(q_pos)) * lattice_constant
            q_to_hole_vec_norm = n.linalg.norm(q_to_hole_vec)
            
            if q_to_hole_vec_norm > 0:
                electric_field_charges += q * q_to_hole_vec/(q_to_hole_vec_norm)**3
                
                #if subfig_label == "a)" and hole_index == 0:
                #    print(q * q_to_hole_vec/(q_to_hole_vec_norm)**3, q/ec, q_pos)
            
        #Dipole electric field
        for (dipole_pos, dipole) in zip(x_O_dipoles, O_dipoles):
            
            dipole_to_hole_vec = n.subtract(n.array(hole_pos), n.array(dipole_pos)) * lattice_constant
            dipole_to_hole_vec_norm = n.linalg.norm(dipole_to_hole_vec)
            
            electric_field_dipoles += n.subtract(3*n.dot(dipole, dipole_to_hole_vec)*dipole_to_hole_vec/(dipole_to_hole_vec_norm)**5, dipole/(dipole_to_hole_vec_norm)**3)
       
        
        total_electric_field = n.add(electric_field_charges, electric_field_dipoles)
        total_electric_field_norm = n.linalg.norm(total_electric_field)
        total_force = (ec+q) * total_electric_field
        total_force_norm = n.linalg.norm(total_force)
        if hole_index == 1:
            pass
            #print(f"{subfig_label} Hole {hole_index}: Ep = {electric_field_dipoles},      Eq = {electric_field_charges},        E(Total) = {total_electric_field}")
            #print(f"           |E(Total)| = {total_electric_field_norm},        F(Total) = {total_force},        |F(Total)| = {total_force_norm}")
    #print("")
        






#%% Figure 4: O-O, Cu-Cu and O-O Screened V'(R) with Varying Parameters

### Figure Creation
multiplier_res = 7
dpi=500
fig = plt.figure(dpi=dpi, figsize=(1*multiplier_res, 1.3*multiplier_res))
spec = gridspec.GridSpec(nrows=3, ncols=2, figure=fig, wspace=0.2 , hspace=0.5)
sub_gridspecs = [[spec[row,col].subgridspec(1,2,wspace=0.1) for col in range(2)] for row in range(3)]
ax_list_antinodal = [[fig.add_subplot(sub_gridspecs[row][col][0]) for col in range(2)] for row in range(3)]
ax_list_nodal = [[fig.add_subplot(sub_gridspecs[row][col][1], sharey=ax_list_antinodal[row][col]) for col in range(2)] for row in range(3)]

xmin = 0
xmax = 4
ymin_list = [[-0.5,-0.5],[-0.5,0],[0,0]]
ymax_list = [[5,4],[3,2],[2,2]]
label_size = 19
tick_size = 12
legend_size = 9
legend_xbox1_list = [[-0.06, -0.06], [-0.09, -0.09], [-0.06, -0.06]]
legend_xbox2_list = [[legend_xbox1_list[0][0] + 0.3, legend_xbox1_list[0][1] + 0.3], [legend_xbox1_list[1][0] + 0.6, legend_xbox1_list[1][1] + 0.3], [legend_xbox1_list[2][0] + 0.3, legend_xbox1_list[2][1] + 0.3]]
legend_ybox_list = [[1.27,1.285], [1.285,1.285], [1.27,1.27]]
label_pad = 0.1
line_width = 3
markersize = 3
yticks_list = [[[0,1,2,3,4], [0,1,2,3,4]],[[0,1,2,3], [0,1,2]],[[0,1,2], [0,1,2]]]
legend_box_pos = [[(-0.15 + legend_xbox1_list[0][0], legend_ybox_list[0][0]-0.22), (-0.15 + legend_xbox1_list[0][1], legend_ybox_list[0][1]-0.235)],[(-0.23 + legend_xbox1_list[1][0],legend_ybox_list[1][0]-0.23), (-0.23 + legend_xbox1_list[1][1], legend_ybox_list[1][1]-0.23)],[(-0.15+ legend_xbox1_list[2][0], legend_ybox_list[2][0]-0.22), (-0.15 + legend_xbox1_list[2][1], legend_ybox_list[2][1]-0.22)]]
legend_box_width = [[8.8, 9.0],[9.0, 9.0],[8.8, 8.8]]
color_list = ['darkblue', "chocolate", "green", "firebrick"]
marker_list = ["o", "^", "s", "*", "D"]
legend_labels = ["$V'$", "Monopole", "Dipole", "Same-Site"]
dist_V0 = n.linspace(0.1, xmax, 100)
V0 = [(ec**2)/(abs(i)*lattice_constant)*6.242e+11 for i in dist_V0 ]
fig.supxlabel('R/a', y = 0.04, fontsize = label_size)
fig.supylabel("$V'$ (eV)", x=0.03, fontsize = label_size)

#Import Data and Parameter Lists
r_0 = 4
max_dist = 8
type_of_energy = "energy"
subtracted_type_of_energy = "energy Madelung"
delta_list = [i for i in delta_array_converged if i not in []]
tdp_list = [i for i in tdp_array_converged if i not in [5.2]]
tpp_list = [i for i in tpp_array_converged if i not in [0.975]]
pol_list = [i for i in pol_array_converged if i not in [1.375]]
legend_label_units = [["$\\Delta^0$", "$t^0_{dp}$"], ["$\\alpha^0_O$","$t^0_{pp}$"], ["$\\Delta^0$", "$\\Delta^0$"]]
variable_list = [["delta_0", "tdp"], ["pol_O","tpp"], ["delta_0", "delta_0"]]
parameters_lists = [[delta_list, tdp_list], [pol_list,tpp_list], [delta_list, delta_list]]
base_parameters = [[delta_0_base, tdp_base], [pol_O_base,tpp_base], [delta_0_base, delta_0_base]]
first_hole_atom_list = [["O","O"],["O","O"],["Cu","O"]]
second_hole_atom_list = [["O","O"],["O","O"],["Cu","Cu"]]

#Looping over subfigs
for row in range(3):
    for col in range(2):
        ax_antinodal = ax_list_antinodal[row][col]
        ax_nodal = ax_list_nodal[row][col]
            
        ### Plotting V0 (bare potential)
        plots_list_legend= [ax_antinodal.plot(dist_V0, V0, color = 'black', linewidth = line_width, zorder = 2)[0]]
        ax_nodal.plot(dist_V0, V0, color = 'black', linewidth = line_width, zorder = 2)
        legend_labels = [r"$V_0$"]
        
        #Hole atom types
        first_hole_atom_string = first_hole_atom_list[row][col]
        second_hole_atom_string = second_hole_atom_list[row][col]
        
        #Base Parameters
        delta_0 = delta_0_base
        tdp = tdp_base
        tpp = tpp_base
        pol_O = pol_O_base
        
        #Importing data files for different parameters
        for param_idx, globals()[variable_list[row][col]] in enumerate(parameters_lists[row][col]):
            with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Two_Holes", max_dist), 'rb') as f:
                data_two_holes = pickle.load(f)
            with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "First_Hole", max_dist), 'rb') as f:
                data_first_hole = pickle.load(f)
            with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Second_Hole", max_dist), 'rb') as f:
                data_second_hole = pickle.load(f)
            with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "No_Holes", max_dist), 'rb') as f:
                data_no_holes = pickle.load(f)
                
            #Hole positions and distance arrays
            first_hole_pos = [0,0] if first_hole_atom_string == "Cu" else [0.5,0]
            second_hole_pos_up = data_two_holes["second hole positions up"]
            second_hole_pos_right = data_two_holes["second hole positions right"]
     
            #Energy of different hole configurations
            energy_two_holes_up = n.array([ data_two_holes[str(i)][type_of_energy] - data_two_holes[str(i)][subtracted_type_of_energy]  for i in second_hole_pos_up ]) 
            energy_two_holes_right = n.array([ data_two_holes[str(i)][type_of_energy] - data_two_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
            energy_first_hole_up = n.array([ data_first_hole[str(i)][type_of_energy] - data_first_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
            energy_first_hole_right = n.array([ data_first_hole[str(i)][type_of_energy] - data_first_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
            energy_second_hole_up = n.array([ data_second_hole[str(i)][type_of_energy] - data_second_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
            energy_second_hole_right = n.array([ data_second_hole[str(i)][type_of_energy] - data_second_hole[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right ]) 
            energy_no_holes_up = n.array([ data_no_holes[str(i)][type_of_energy] - data_no_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_up ]) 
            energy_no_holes_right = n.array([ data_no_holes[str(i)][type_of_energy] - data_no_holes[str(i)][subtracted_type_of_energy] for i in second_hole_pos_right]) 
            energy_subtracted_up = n.array([energy_two_holes_up[i] - energy_first_hole_up[i] - energy_second_hole_up[i] + energy_no_holes_up[i] for i,_ in enumerate(energy_two_holes_up)])
            energy_subtracted_right = n.array([energy_two_holes_right[i] - energy_first_hole_right[i] - energy_second_hole_right[i] + energy_no_holes_right[i] for i,_ in enumerate(energy_two_holes_right)])

            #Plotting V' (screened potential)
            legend_label = r"%.2f" % (globals()[variable_list[row][col]]/base_parameters[row][col]) + legend_label_units[row][col]
            energy_subtracted_antinodal = [i for idx,i in enumerate(energy_subtracted_right) if second_hole_pos_right[idx][1] == 0.0]
            dist_antinodal = [i[0]-0.5 for i in second_hole_pos_right if i[1] == 0.0]
            if row < 2:
                dist_nodal = [i[1] for i in second_hole_pos_up if i[0]-0.5 == 0.0]
                energy_subtracted_nodal = [i for idx,i in enumerate(energy_subtracted_up) if second_hole_pos_up[idx][0]-0.5 == 0.0]
            else:
                if col == 0:
                    dist_nodal = [i[1]-0.5 for i in second_hole_pos_up if i[0] == 0.0]
                    energy_subtracted_nodal = [i for idx,i in enumerate(energy_subtracted_up) if second_hole_pos_up[idx][0] == 0.0]
                elif col == 1:
                    dist_nodal = [i[1] for i in second_hole_pos_up if i[0] - 1.0 == 0.0]
                    energy_subtracted_nodal = [i for idx,i in enumerate(energy_subtracted_up) if second_hole_pos_up[idx][0] - 1.0 == 0.0]
            plots_list_legend.append(ax_antinodal.plot(dist_antinodal, energy_subtracted_antinodal, linewidth=line_width, color=color_list[param_idx], marker = marker_list[param_idx], label=legend_label)[0])
            ax_nodal.plot(dist_nodal, energy_subtracted_nodal, linewidth=line_width, color=color_list[param_idx], marker = marker_list[param_idx], label=legend_label)
            legend_labels.append(legend_label)
            

        #Legends
        legend_1 = ax_antinodal.legend(plots_list_legend[0:3], legend_labels[0:3], loc = "upper left", bbox_to_anchor=(legend_xbox1_list[row][col],legend_ybox_list[row][col]), fontsize = legend_size, ncols=3, edgecolor="white", borderaxespad=0, framealpha=0)
        legend_2 = ax_antinodal.legend(plots_list_legend[3:], legend_labels[3:], loc="upper left", bbox_to_anchor=(legend_xbox2_list[row][col],legend_ybox_list[row][col]-0.1), fontsize = legend_size, ncols = len(legend_labels)-3, edgecolor="white", borderaxespad=0, framealpha=0)
        ax_antinodal.add_artist(legend_1)
        ax_antinodal.add_patch(patches.Rectangle(legend_box_pos[row][col], width=legend_box_width[row][col], height=0.21, linewidth=1.5, transform=ax_antinodal.get_xaxis_transform(), edgecolor="black", facecolor="none", clip_on=False))


        #Ticks and Limits
        ax_nodal.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_antinodal.tick_params(axis="both", labelsize=tick_size)
        ax_nodal.tick_params(axis="both", labelsize=tick_size)
        ax_antinodal.set_ylim(ymin_list[row][col],ymax_list[row][col])
        ax_antinodal.set_yticks(yticks_list[row][col])
        ax_antinodal.set_xlim(xmin,xmax)
        ax_nodal.set_xlim(xmin,xmax)
        ax_antinodal.set_xticks([0,2,4])
        ax_nodal.set_xticks([0,2,4])
        ax_antinodal.grid(zorder=-10)
        ax_nodal.grid(zorder=-10)
        # if row < 2:
        #     ax_antinodal.sharex(ax_list_antinodal[2][col])
        #     ax_nodal.sharex(ax_list_nodal[2][col])
        #     ax_antinodal.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        #     ax_nodal.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


    

















#%% Supplementary Figure: Single Real Space Polarization Plotting

#### Figure Creation Settings
fig_width = 3.375 * 0.75
fig_height = fig_width 
dpi = 800
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), dpi=dpi)
    
#Global sizes of plot elements
xlim = 2
ylim = 2
axis_label_fontsize = 15
tick_label_fontsize = 10
legend_fontsize = 5
line_width = 1.5
line_markersize = 2.2



#### Import Parameters
r_0 = 4
max_dist = 4
type_of_energy = "energy"
configuration = "no holes"
first_hole_atom_string = "Cu"
second_hole_atom_string = "Cu"
second_hole_pos = [0,0]
first_hole_pos = [0,0] if first_hole_atom_string == "Cu" else [0.5,0.0]
#first_hole_pos = [0.5,0.0] if first_hole_atom_string == "Cu" else [0.5,0.0]
hole_positions = [first_hole_pos, second_hole_pos]

#Base Parameters
delta_0 = 4.5
tdp = tdp_base
tpp = tpp_base
tpp = tpp_base
pol_O = pol_O_base

#Importing data files
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Two_Holes", max_dist), 'rb') as f:
    data_two_holes = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "First_Hole", max_dist), 'rb') as f:
    data_first_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "Second_Hole", max_dist), 'rb') as f:
    data_second_hole = pickle.load(f)
with open("charge_transfer_normalized//from_doping_calcs//exact_2_by_2//monopole//r4//new_delta//r0_%d_tdp_%.2f_tpp_%.2f_Opol_%.2f_delta_%.2f_firstHole_%s_secondHole_%s_%s_maxDist_%d.pkl" % (r_0, tdp,tpp,pol_O, delta_0, first_hole_atom_string, second_hole_atom_string, "No_Holes", max_dist), 'rb') as f:
    data_no_holes = pickle.load(f)


#### Plotting Settings
#Plotting limits, and making sure the x range and y range are equal
max_dist_from_hole = 2
xlims = [min([hole_pos[0] for hole_pos in hole_positions]) - max_dist_from_hole, max([hole_pos[0] for hole_pos in hole_positions]) + max_dist_from_hole]
ylims = [min([hole_pos[1] for hole_pos in hole_positions]) - max_dist_from_hole, max([hole_pos[1] for hole_pos in hole_positions]) + max_dist_from_hole]
xrange = xlims[1] - xlims[0]
yrange = ylims[1] - ylims[0]
if yrange > xrange:
    xlims[0] = xlims[0] + (xrange-yrange)/2
    xlims[1] = xlims[1] - (xrange-yrange)/2
else:
    ylims[0] = ylims[0] + (yrange-xrange)/2
    ylims[1] = ylims[1] - (yrange-xrange)/2
    

#Removing axis ticks and setting limits
lim_offset = 0.18
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both',bottom=False, top=False, left=False, labelbottom=False, labelleft=False)    
ax.set_xlim(xlims[0] + lim_offset, xlims[1] - lim_offset)
ax.set_ylim(ylims[0] + lim_offset, ylims[1] - lim_offset)

#Adding background colours to represent Paths 1 and 2 (done by plotting three points as triangles, and dashed lines to separate regions)
Path_1_Triangle_up = n.array([ hole_positions[0], n.add(hole_positions[0], [2,2]), n.add(hole_positions[0], [-2,2]), hole_positions[0]])
Path_1_Triangle_down = n.array([ hole_positions[0], n.add(hole_positions[0], [2,-2]), n.add(hole_positions[0], [-2,-2]), hole_positions[0]])
Path_2_Triangle_right = n.array([ hole_positions[0], n.add(hole_positions[0], [2,2]), n.add(hole_positions[0], [2,-2]), hole_positions[0]])
Path_2_Triangle_left = n.array([ hole_positions[0], n.add(hole_positions[0], [-2,2]), n.add(hole_positions[0], [-2,-2]), hole_positions[0]])
paths_triangles = n.array([Path_1_Triangle_up, Path_1_Triangle_down, Path_2_Triangle_right, Path_2_Triangle_left])
triangle_color_array = ["#85ffffff", "#85ffffff", "#dbc5ffff", "#dbc5ffff"]

for (triangle_idx, triangle_points) in enumerate(paths_triangles):
    ax.plot( [i[0] for i in triangle_points], [i[1] for i in triangle_points], linestyle = "",  zorder=-2) #Region 1 (up)
    ax.fill( [i[0] for i in triangle_points], [i[1] for i in triangle_points], linestyle = "", facecolor = triangle_color_array[triangle_idx], alpha = 0.5, zorder=-2) #Region 1 (up)
 

#### Importing the right data based on the chosen configuratoin
if configuration == "two holes":
    energy_results = data_two_holes
    dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
    mono_charges = energy_results[str(second_hole_pos)]["charges"]
    dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )

elif configuration == "first hole":
    hole_positions = hole_positions[0:1]
    energy_results = data_first_hole
    dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
    mono_charges = energy_results[str(second_hole_pos)]["charges"]
    dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )

elif configuration == "second hole":
    hole_positions = hole_positions[1:2]
    energy_results = data_second_hole
    dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
    mono_charges = energy_results[str(second_hole_pos)]["charges"]
    dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )

elif configuration == "no holes":
    hole_positions = []
    energy_results = data_no_holes
    dipoles = n.hstack(energy_results[str(second_hole_pos)]["O dipoles"])
    mono_charges = energy_results[str(second_hole_pos)]["charges"]
    dipoles = n.hstack( [ d for (i,d) in enumerate(energy_results[str(second_hole_pos)]["O dipoles"]) if mono_charges[i][1] not in hole_positions ] )


elif configuration == "sum of holes 1 and 2":
    energy_results = data_second_hole
    mono_charges = [ [(q1[0] + q2[0])/2, q1[1]] if q1[1] not in hole_positions else [0, q1[1]] for (q1,q2) in zip(data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"])]
    dipoles = n.add( data_first_hole[str(second_hole_pos)]["O dipoles"], data_second_hole[str(second_hole_pos)]["O dipoles"] )/2
    dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])

elif configuration == "subtracted":
    energy_results = data_two_holes
    mono_charges = [ [q_two_holes[0] - q_first_hole[0] - q_second_hole[0] + q_no_holes[0], q_two_holes[1]] for (q_two_holes,q_first_hole,q_second_hole,q_no_holes) in zip(data_two_holes[str(second_hole_pos)]["charges"], data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"], data_no_holes[str(second_hole_pos)]["charges"])]
    dipoles = n.add( n.subtract( n.subtract( data_two_holes[str(second_hole_pos)]["O dipoles"], data_first_hole[str(second_hole_pos)]["O dipoles"] ), data_second_hole[str(second_hole_pos)]["O dipoles"]), data_no_holes[str(second_hole_pos)]["O dipoles"] )
    dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])

elif configuration == "individual holes subtracted":
    energy_results = data_first_hole
    mono_charges = [ [q_first_hole[0] + q_second_hole[0] - 2*q_no_holes[0], q_first_hole[1]] for (q_first_hole,q_second_hole,q_no_holes) in zip(data_first_hole[str(second_hole_pos)]["charges"], data_second_hole[str(second_hole_pos)]["charges"], data_no_holes[str(second_hole_pos)]["charges"])]
    dipoles = n.subtract( n.subtract( n.add( data_first_hole[str(second_hole_pos)]["O dipoles"], data_second_hole[str(second_hole_pos)]["O dipoles"] ), data_no_holes[str(second_hole_pos)]["O dipoles"]), data_no_holes[str(second_hole_pos)]["O dipoles"] )
    dipoles = n.hstack([d for (i,d) in enumerate(dipoles) if mono_charges[i][1] not in hole_positions])

#Imported Position Arrays
x_O = n.array(energy_results[str(second_hole_pos)]["x_O"])
x_O_dipoles = n.array([x for x in x_O if x.tolist() not in hole_positions])
x_Cu = n.array(energy_results[str(second_hole_pos)]["x_Cu"])

#### Plotting Coloured Dots
#O and Cu Sites
for i in range(len(x_O)):
    ax.scatter(x_O[i][0], x_O[i][1], color='#dd1308ff')
for i in range(len(x_Cu)):
    ax.scatter(x_Cu[i][0], x_Cu[i][1], color='#8b4513ff', s=11, zorder=2)

#+1 Doped Holes
for hole_pos in hole_positions:
    ax.scatter(hole_pos[0], hole_pos[1], color='#05ca1cff', s = 100, marker = "x")
    ax.annotate("+" + str(hole_positions.count(hole_pos)), xy=(hole_pos[0],hole_pos[1]), fontsize=5.5, color = "#ffff00ff", weight='bold', ha="center", va="center", path_effects=[pe.withStroke(linewidth=3, foreground="black")])


###Plotting the O dipoles
quiver_scale_O = max(dipoles[0:-1:2])*7
#quiver_scale_O = 1.8E-17
print(quiver_scale_O)
ax.quiver(x_O_dipoles.T[0], x_O_dipoles.T[1], dipoles[0:-1:2], dipoles[1:len(dipoles):2], color = "#00ff00ff", edgecolor = "black", scale = quiver_scale_O, pivot="mid", label="O Dipoles", headlength = 5.5*1.4, headaxislength = 5.5, headwidth = 5.5, linewidth = 1, width = 0.01, zorder = 2)

###Writing the induced charges
charge_fontsize = 10
for q_idx, (q, bond_pos) in enumerate(mono_charges):
    if int(bond_pos[0] + bond_pos[1]) != (bond_pos[0] + bond_pos[1]):
        #color = "#ad0e98ff"
        color = "orange"
    else:
        #color = "#0b3db8ff"
        color = "dodgerblue"
        
    if xlims[0] < bond_pos[0] < xlims[1] and ylims[0] < bond_pos[1] < ylims[1]:
        ax.text(bond_pos[0], bond_pos[1] + 0.26*0.6, "%d" % (q/ec*101), fontsize = charge_fontsize,  color = color, weight="heavy", path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],  verticalalignment = "center", horizontalalignment = "center" , zorder = 6)








#%% G vector count

a = 5
dist = 1

max_n = int(n.ceil(a/dist)) * 2

number_g_vects = 0

g_vector_list = []

'''        
#2D
for nx_1 in range(-max_n, max_n+1):
    for nx_2 in range(-max_n, max_n+1):
        for ny_1 in range(-max_n, max_n+1):
            for ny_2 in range(-max_n, max_n+1):
        
                if n.sqrt( (nx_1 + nx_2)**2 + (ny_1 + ny_2)**2) <= a/dist:
                    
                    if (nx_1, ny_1) not in g_vector_list:
                        g_vector_list.append((nx_1, ny_1))
                        
                    if (nx_2, ny_2) not in g_vector_list:
                        g_vector_list.append((nx_2, ny_2))
                        
'''

#3D
for nx_1 in range(-max_n, max_n+1):
    for nx_2 in range(-max_n, max_n+1):
        for ny_1 in range(-max_n, max_n+1):
            for ny_2 in range(-max_n, max_n+1):
                for nz_1 in range(-max_n, max_n+1):
                    for nz_2 in range(-max_n, max_n+1):
        
                        if n.sqrt( (nx_1 + nx_2)**2 + (ny_1 + ny_2)**2 + (nz_1 + nz_2)**2) <= a/dist:
                            
                            if (nx_1, ny_1, nz_1) not in g_vector_list:
                                g_vector_list.append((nx_1, ny_1, nz_1))
                                
                            if (nx_2, ny_2, nz_2) not in g_vector_list:
                                g_vector_list.append((nx_2, ny_2, nz_2))
                        
number_g_vects = len(g_vector_list)

            
print("For a = %.2f and d = %.2f =  %.2fa, there are %d G-vectors in 2D." % (a, dist, dist/a, number_g_vects))














