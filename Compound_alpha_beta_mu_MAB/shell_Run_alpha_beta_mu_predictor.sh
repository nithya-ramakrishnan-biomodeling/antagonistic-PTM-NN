#!/bin/bash
# To run the alpha-beta-mu predictor simulation and Create Figures 5.1, 5.2, and 5.3 in the paper.
start_time=$(date +%s)

trap "echo 'Interrupted. Killing background jobs...'; jobs -p | xargs -r kill; exit 1" SIGINT

python3 ../Run_alpha_beta_mu_predictor.py

wait

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total time for simulation: ${elapsed} seconds"
