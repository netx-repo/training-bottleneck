
profile:
	nvcc profile.cu -o profile

gen_sizes:
	python3 gen_layer_sizes.py

run: profile gen_sizes
	python3 run_mem_profile.py

summarize:
	python3 parse_log.py

bw: summarize
	python3 memcpy_bw.py