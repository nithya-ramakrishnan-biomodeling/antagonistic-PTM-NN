alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
betas="[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]"
mu=0.5  # Set mu as needed
gamma=0.5  # Set gamma as needed

start_time=$(date +%s)

trap "echo 'Interrupted. Killing background jobs...'; jobs -p | xargs -r kill; exit 1" SIGINT

for alpha in "${alphas[@]}"; do
    batch_name="Targeted_MA_DABC_mu${mu}_gamma${gamma}_alpha${alpha}"
    python3 -u ../Run_ESP_ABC_Simple.py \
        --batch_name "${batch_name}" \
        --alpha_eval "[${alpha}]" \
        --beta_eval "${betas}" \
        --mu ${mu} \
        --gamma ${gamma} \
        --seed 42 \
        --seq_length 100 \
        --n_samples 100000 \
        --n_samples_test 30000 \
        --input_size 300 \
        --hidden_size 150 \
        --output_size 100 \
        --num_layers 5 \
        --learning_rate 0.001 \
        --num_epochs 10000 \
        --batch_size 100000 \
        --loss_function_type MSE \
        --verbose_level 2 \
        --visualise_nn False &
done

wait

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total time for all parallel simulations: ${elapsed} seconds"

# Concatenate all Targeted_MA_DABC_mu${mu}_gamma${gamma}_*_BitError_Combinations.csv files into one, keeping only the first header
output_file="Targeted_MA_DABC_mu${mu}_gamma${gamma}_BitError_Combinations.csv"
first=1
> "$output_file"
for alpha in "${alphas[@]}"; do
    file="Targeted_MA_DABC_mu${mu}_gamma${gamma}_alpha${alpha}_BitError_Combinations.csv"
    if [ -f "$file" ]; then
        if [ $first -eq 1 ]; then
            cat "$file" >> "$output_file"
            first=0
        else
            tail -n +2 "$file" >> "$output_file"
        fi
    fi
done
echo "Combined CSV written to $output_file"

python3 ../batch_matrix_generator.py Targeted_MA_DABC_mu${mu}_gamma${gamma}

