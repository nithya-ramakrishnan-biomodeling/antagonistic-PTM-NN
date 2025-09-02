# To Recreate the figure 4.1 in the paper, run this script.
start_time=$(date +%s)

trap "echo 'Interrupted. Killing background jobs...'; jobs -p | xargs -r kill; exit 1" SIGINT

python3 -u ../Run_ESP_range_Antagonist_Simple.py
# wait

# This is not enough to generate 4.1 as we also need to use the trained network weight to check for each alpha beta pair.
python -u ../Run_Single_Network_Heatmap.py --batch_name Range_Test_Simple_heatmap --weight_path Range_Test_Simple/network_weights.pth

cd Range_Test_Simple_heatmap
# for mu in 0.0 to 0.9, cd into the mu folder mu_0_0 to mu_0_9
# pass the parameter Range_Test_Simple_heatmap_mu_a_b to the batch matrix generator
for mu in $(seq 0.0 0.1 0.9)
do
    filename="mu_${mu//./_}/Range_Test_Simple_heatmap_mu_${mu//./_}"
    echo "Generating batch matrix for ${filename}"
    python3 ../../batch_matrix_generator.py ${filename}
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total time for simulation: ${elapsed} seconds"