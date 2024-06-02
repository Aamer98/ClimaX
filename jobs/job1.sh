#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=climax_job1
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=19000M
#SBATCH --time=0-20:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/my_env/climaX/bin/activate
wandb offline

echo "------------------------------------< Data preparation>----------------------------------"
date +"%T"
cd $SLURM_TMPDIR

echo "Copying the datasets"
date +"%T"
mkdir climate_bench
cd climate_bench
cp /home/aamer98/projects/def-ebrahimi/aamer98/data/climate_bench/climate_bench.zip .
unzip climate_bench.zip
rm climate_bench.zip


echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"

cd $SLURM_TMPDIR
cp -r /home/aamer98/projects/def-ebrahimi/aamer98/repos/ClimaX .
cd ClimaX

python src/climax/climate_projection/train.py --config configs/climate_projection.yaml --trainer.strategy=ddp --trainer.devices=2 --trainer.max_epochs=50 --data.root_dir=$SLURM_TMPDIR/climate_bench/5.625deg --data.batch_size=8 --model.pretrained_path='/home/aamer98/projects/def-ebrahimi/aamer98/repos/ClimaX/checkpoints/5.625deg.ckpt' --model.lr=5e-4 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5


echo "----------------------------------------<End of program>------------------------------------"
date +"%T"