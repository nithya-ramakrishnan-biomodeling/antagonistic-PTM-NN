# To Recreate the figure 4.1 in the paper, run this script.
start_time=$(date +%s)

trap "echo 'Interrupted. Killing background jobs...'; jobs -p | xargs -r kill; exit 1" SIGINT

python3 -u ../Run_ESP_range_Antagonist_Simple.py

wait

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total time for simulation: ${elapsed} seconds"

# This is not enough to generate 4.1 as we also need to use the trained network weight to check for each alpha beta pair.