# To Recreate the figure 3.1 in the paper, run this script. But using grid search to find the best hyperparameters
# The hyper parameters we try to optimize are :: hidden size, num_layers, learning rate
# we create a folder for each combination of hyperparameters and within each run the simulations to produce the results.
hidden_sizes=(50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
num_layers=(1 2 3 4 5 6 7 8 9 10)
learning_rates=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)

mkdir "gridsearch_results"

for hidden_size in "${hidden_sizes[@]}"; do
    for num_layer in "${num_layers[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            # Create a directory for this combination of hyperparameters
            dir_name="gridsearch_hidden_${hidden_size}_layers_${num_layer}_lr_${learning_rate}"
            mkdir -p "$dir_name"
            cd "$dir_name" # This takes us into the simulation directory

            alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
            betas="[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]"

            start_time=$(date +%s)

            trap "echo 'Interrupted. Killing background jobs...'; jobs -p | xargs -r kill; exit 1" SIGINT

            for alpha in "${alphas[@]}"; do
                batch_name="Targeted_MA_DA_${alpha}"
                python3 -u ../../Run_ESP.py \
                --batch_name ${batch_name} \
                --alpha_eval [${alpha}] \
                --beta_eval ${betas} \
                --seed 42 \
                --seq_length 100 \
                --n_samples 100000 \
                --n_samples_test 30000 \
                --input_size 100 \
                --hidden_size $hidden_size \
                --output_size 100 \
                --num_layers $num_layer \
                --learning_rate $learning_rate \
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

            # Concatenate all Targeted_MA_DA_*_BitError_Combinations.csv files into one, keeping only the first header
            output_file="Targeted_MA_DA_BitError_Combinations.csv"
            first=1
            > "$output_file"
            for alpha in "${alphas[@]}"; do
                file="Targeted_MA_DA_${alpha}_BitError_Combinations.csv"
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

            python3 ../../batch_matrix_generator.py Targeted_MA_DA

            cd .. # This should bring us back to the Targeted_MA_DA directory

            mv "$dir_name" "gridsearch_results/"
        done
    done
done
