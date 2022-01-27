#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=ImageNet_na
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-10:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load python/3.7
module load cuda cudnn 
source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/meta-dataset

cp ~/scratch/imagenet_object_localization_patched2019.tar.gz .

tar -xzf imagenet_object_localization_patched2019.tar.gz


cp ~/scratch/metadatasets/wordnet.is_a.txt $SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/train/
cp ~/scratch/metadatasets/wordnet.is_a.txt $SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/train/

cd meta-dataset

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=ilsvrc_2012 \
  --ilsvrc_2012_data_root=$SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/train/ \
  --splits_root=SPLITS \
  --records_root=RECORDS

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/meta-dataset/RECORDS/ ~/scratch/metadatasets/