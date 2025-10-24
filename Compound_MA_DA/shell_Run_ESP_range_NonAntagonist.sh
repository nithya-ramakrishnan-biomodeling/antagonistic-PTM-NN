# To Recreate the figure 4.1 in the paper, run this script.
start_time=$(date +%s)

trap "echo 'Interrupted. Killing background jobs...'; jobs -p | xargs -r kill; exit 1" SIGINT

python3 -u ../Run_ESP_range_No_Antagonism.py
# wait

python -u ../Run_Single_Network_Heatmap.py --batch_name Range_Test_No_Anta_heatmap --weight_path Range_Test_No_Anta/network_weights.pth

cd Range_Test_No_Anta_heatmap
# for mu in 0.0 to 0.9, cd into the mu folder mu_0_0 to mu_0_9
# pass the parameter Range_Test_No_Anta_heatmap_mu_a_b to the batch matrix generator
for mu in $(seq 0.0 0.1 0.9)
do
    filename="mu_${mu//./_}/Range_Test_No_Anta_heatmap_mu_${mu//./_}"
    echo "Generating batch matrix for ${filename}"
    python3 ../../batch_matrix_generator.py ${filename}
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total time for simulation: ${elapsed} seconds"