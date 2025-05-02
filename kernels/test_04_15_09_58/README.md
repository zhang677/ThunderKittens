Modify the Makefile of 4090.cu on Runpod to align the python version with python-config version.

pip install pybind11

ncu --export ncu_report -f --set full --target-processes all python bench.py

ncu --launch-skip 4 --launch-count 1 --export ncu_report -f --set full --target-processes all python run_once.py 13 8 16 512 64

ncu --import ncu_report.ncu-rep --kernel-name "attend_ker" --csv --page details --log-file output_details.csv

python filter_output.py output_details.csv output_attend_ker.csv

./sweep_seqlen.sh
Note: this script appends to the output latency file.

python plot_scripts/plot_latency.py profile_results/output.csv profile_results/output.png

python plot_scripts/plot_multi_latency.py profile_results/four_outputs.png profile_results/output_17x8x16x64.csv profile_results/output_17x8x16x128.csv profile_results/output_17x8x32x64.csv profile_results/output_17x8x32x128.csv

python plot_scripts/plot_multi_latency.py profile_results/four_simulations.png profile_results/simulated_17x8x16x64.csv profile_results/simulated_17x8x16x128.csv profile_results/simulated_17x8x32x64.csv profile_results/simulated_17x8x32x128.csv

python plot_scripts/plot_latency_compare.py profile_results/four_compare.png profile_results/output_17x8x16x64.csv profile_results/output_17x8x16x128.csv profile_results/output_17x8x32x64.csv profile_results/output_17x8x32x128.csv

python plot_scripts/plot_rel_error.py profile_results/four_relative.png profile_results/output_17x8x16x64.csv profile_results/output_17x8x16x128.csv profile_results/output_17x8x32x64.csv profile_results/output_17x8x32x128.csv

python plot_scripts/plot_16_rel_error.py profile_results/16_relative.png

python plot_scripts/plot_16_reg.py profile_results/16_regs.png

python plot_scripts/plot_rel_error_heatmap.py profile_results/16_relative_heatmap.png

python plot_scripts/plot_16_steady.py d a100 profile_results_a100/16_steady_d.png

python plot_scripts/plot_16_steady.py d l40s profile_results_l40s/16_steady_d.png

python plot_scripts/plot_16_steady_fit.py d a100 profile_results_a100/16_fit_d.png

python plot_scripts/plot_util.py profile_results_a100 a100

python plot_scripts/plot_16_steady_roofline.py d a100 profile_results_a100/16_steady_d_roofline.png

python plot_scripts/plot_single_roofline.py profile_results_a100/short_13x8x32x6336x160.csv a100 profile_results_a100/plot_roofline_13x8x32x6336x160.png

export PYTHONPATH=$PYTHONPATH:/usr/local/cuda/nsight-compute-2023.1.1/extras/python

./profile_once_l40s.sh 16 396 64