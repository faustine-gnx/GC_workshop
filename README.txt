For medial motor-correlated neurons of example plane:
background: mean background 
cell_centers: position of neuron centers on background [x, y]
example_HB_calcium_traces: fluorescence traces of neurons
tail_angle: tail_angle over time
xaxes_and_stim:


Complete hindbrain data Fish 6 Trial 07: df_hindbrain_F6T07.pkl (pandas dataframe)

Columns:
fluo: fluorescence traces [n_cells x n_timesteps]
cell_centers: x and y position of the cell center [n_cells x 2]
background: plane background for plotting [249 x 512]
n_cells: number of cells in the plane
tail_angle: array of angle of the tail [75000,] - 75000 timesteps: higher frequency than calcium imaging recording
tail_angle_regressor: tail angle convolved to calcium decay function [75000,]
is_swim: boolean array whether each cell in correlated to swim activity (True if pearson correlation between cell's fluorescence trace and tail_angle_regressor > 0.6) [n_cells,]
swim_neurons: indices of swim-correlated neurons [n_swim_cells,]
medial_neurons: indices of swim-correlated neurons in the medial region [n_medial_cells,]
only_swim_neurons: indices of swim-correlated neurons NOT in the medial region [n_only_swim_cells,]
SNR: signal-to-noise ratio for each cell [n_cells,]
BV_GC_all: original bivariate (BV) Granger causality results matrix on all cells [n_cells,n_cells]
BV_GC_medial: original bivariate (BV) Granger causality results matrix [n_medial_cells,n_medial_cells]
BV_Fstat_medial: original BV F-statistics matrix [n_medial_cells,n_medial_cells]
BV_threshold_F_ori_all: original threshold for the BV F-statistics significance for GC on all cells.
BV_threshold_F_ori: original threshold for the BV F-statistics significance for GC on medial cells only. /!\ BV_threshold_F_ori and BV_threshold_F_ori_all different due to Bonferroni correction
BV_threshold_F_new_mat_medial: new threshold customized for each pair of neurons (BV)  [n_medial_cells,n_medial_cells]
BV_Fstat_normalized_medial: new BV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
BV_GC_normalized_medial: new BV GC results matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
MV_GC_medial: original multivariate (MV) Granger causality results matrix [n_medial_cells,n_medial_cells]
MV_Fstat_medial: original MV F-statistics matrix [n_medial_cells,n_medial_cells]
MV_threshold_F_ori_medial: original threshold for the MV F-statistics significance
MV_threshold_F_new_mat_medial: new MV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
MV_Fstat_normalized_medial: new MV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
MV_GC_normalized_medial: new MV GC results matrix normalized by customized threshold [n_medial_cells,n_medial_cells]