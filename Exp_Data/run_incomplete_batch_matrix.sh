# modifications = ['H2AK5ac','H2AS129ph','H3K14ac','H3K18ac','H3K23ac','H3K27ac','H3K36me','H3K36me2','H3K36me3','H3K4ac','H3K4me','H3K4me2','H3K4me3','H3K56ac','H3K79me','H3K79me3','H3K9ac','H3S10ph','H4K12ac','H4K16ac','H4K20me','H4K5ac','H4K8ac','H4R3me','H4R3me2s','Htz1']
# modification_time_list = ['t0', 't4', 't8', 't15', 't30', 't60']
# for each time list, for each modification, cd into the folder and run the batch_matrix_incomplete script

modification_time_list=('t0' 't4' 't8' 't15' 't30' 't60')
modifications=('H2AK5ac' 'H2AS129ph' 'H3K14ac' 'H3K18ac' 'H3K23ac' 'H3K27ac' 'H3K36me' 'H3K36me2' 'H3K36me3' 'H3K4ac' 'H3K4me' 'H3K4me2' 'H3K4me3' 'H3K56ac' 'H3K79me' 'H3K79me3' 'H3K9ac' 'H3S10ph' 'H4K12ac' 'H4K16ac' 'H4K20me' 'H4K5ac' 'H4K8ac' 'H4R3me' 'H4R3me2s' 'Htz1')

for modification_time in "${modification_time_list[@]}"; do
    for mod in "${modifications[@]}"; do
        echo "Processing $mod at time $modification_time"
        target_dir="${modification_time}/${mod}_${modification_time}"
        if [ -d "$target_dir" ]; then
            cd "$target_dir" || { echo "Failed to navigate to $target_dir"; exit 1; }
            python3 ../../../batch_matrix_incomplete.py "${mod}_${modification_time}"
            cd ../../ || { echo "Failed to navigate back to root directory"; exit 1; }
        else
            echo "Directory $target_dir does not exist. Skipping..."
        fi
    done
done

