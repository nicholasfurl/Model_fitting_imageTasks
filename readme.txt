Data files for the collected datasets are stored in folders labelled by dataset name, for example `face_study_1_av`.

The `imageTask_RFX_BMS_out` folder contains output from the random-effects Bayesian model selection analysis produced by `imageTasks_RFX_BMS_v2.m`. Tables S1 and S2 are derived from these outputs.

The `outputs` folder contains temporary files and processed data files saved by the parameter-recovery and model-fitting code. This includes output from `imageTask_paper_parameter_recovery_combinedModel.m` and from the model-fitting scripts.

Parameter-recovery analyses depend on `analyzeSecretaryPR.m`, which runs the probabilistic models, including the cost to sample model, biased prior model, and related variants.

For model fitting, `imagetask_master_script.m` is the high-level script that specifies which models to run. The `fit_models*.m` scripts fit individual datasets. These scripts mainly handle dataset-specific input/output and setup, and they call `run_io.m` to add the ground truth / ideal observer benchmark for comparison. They also depend on `imageTask_run_models.m`, which is the main engine for fitting the computational models.

`imageTask_trust2_populations.m` reproduces Figure S8, which compares response distributions over sequence positions for participants best fit by the cost to sample model and participants best fit by the biased prior model in trustworthiness dataset 2.

`imageTasks_figures_JEPLMC_v4_greyscale_weights_use.m` produces most of the paper and supplementary results figures using model-fit data stored in `outputs`.

`imageTasks_position_LL_trust2_v3_two_panel.m` performs the likelihood-by-sequence-position analysis reported in Supplementary Figure S9.

Note that several scripts contain hard-coded local paths and may need to be edited before running on another machine.
