#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/work/firoozh1/nonar/image_captioner/logs/%J_slurm.out
#SBATCH --error=/scratch/work/firoozh1/nonar/image_captioner/logs/%J_slurm.err

export HF_HOME="/scratch/work/firoozh1/nonar/.cache/hugginface/"
export HF_DATASETS_CACHE="/scratch/work/firoozh1/nonar/.cache/hugginface/" 

module load miniconda
echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."
cd $WRKDIR/nonar/image_captioner/
source activate /scratch/work/firoozh1/.conda_envs/img2cap/
python3 scripts/caption_with_blip.py