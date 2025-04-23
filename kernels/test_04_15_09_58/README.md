Modify the Makefile of 4090.cu on Runpod to align the python version with python-config version.

pip install pybind11

ncu --export ncu_report -f --set full --target-processes all python bench.py

ncu --import ncu_report.ncu-rep --kernel-name "attend_ker" --csv --page details --log-file output_details.csv

python filter_output.py output_details.csv output_attend_ker.csv

./sweep_seqlen.sh
Note: this script appends to the output latency file.

python plot_scripts/plot_latency.py profile_results/output.csv profile_results/output.png

python plot_scripts/plot_multi_latency.py profile_results/four_outputs.png profile_results/output_17x8x16x64.csv profile_results/output_17x8x16x128.csv profile_results/output_17x8x32x64.csv profile_results/output_17x8x32x128.csv

python plot_scripts/plot_multi_latency.py profile_results/four_simulations.png profile_results/simulated_17x8x16x64.csv profile_results/simulated_17x8x16x128.csv profile_results/simulated_17x8x32x64.csv profile_results/simulated_17x8x32x128.csv

python plot_scripts/plot_latency_compare.py profile_results/four_compare.png profile_results/output_17x8x16x64.csv profile_results/output_17x8x16x128.csv profile_results/output_17x8x32x64.csv profile_results/output_17x8x32x128.csv

python plot_scripts/plot_rel_error.py profile_results/four_relative.png profile_results/output_17x8x16x64.csv profile_results/output_17x8x16x128.csv profile_results/output_17x8x32x64.csv profile_results/output_17x8x32x128.csv