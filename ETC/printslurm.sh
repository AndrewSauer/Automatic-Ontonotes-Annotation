ls | grep slurm- | xargs -I {} python3 printresult.py {}
python3 statistics_to_csv.py
