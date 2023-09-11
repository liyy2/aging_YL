#!/bin/bash

#SBATCH --job-name=process_data       # Job name
#SBATCH --ntasks=8                        # Run on a single CPU (though you're still using multiprocessing in your script)
#SBATCH --mem=100gb                        # Memory limit
#SBATCH --time=24:00:00                   # Time limit hrs:min:sec
#SBATCH --partition=pi_gerstein               # Partition, or queue, to submit to

eval "$(conda shell.bash hook)"
conda activate /gpfs/gibbs/pi/gerstein/yl2428/torch_geom
export PYTHONPATH=/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/code 
python /gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/code/scGNN/prepare_data.py --num_workers 8 --out_path /gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/lmdb-3 