for i in $(seq 1 30);
do
    sbatch ./run_test.sh $i False
    sbatch ./run_test.sh $i True
done
