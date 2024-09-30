cd baseline

# Sampling with temperature=1

python generate.py --dataset entailmentbank --split train      --sample --temperature 1.0 --sample_size 16
python generate.py --dataset entailmentbank --split validation --sample --temperature 1.0 --sample_size 16

python generate.py --dataset enwn           --split validation --sample --temperature 1.0 --sample_size 16

python generate.py --dataset eqasc          --split train      --sample --temperature 1.0 --sample_size 16
python generate.py --dataset eqasc          --split validation --sample --temperature 1.0 --sample_size 16

python generate.py --dataset folio          --split train      --sample --temperature 1.0 --sample_size 16
python generate.py --dataset folio          --split validation --sample --temperature 1.0 --sample_size 16

python generate.py --dataset prontoqa       --split train      --sample --temperature 1.0 --sample_size 16
python generate.py --dataset prontoqa       --split validation --sample --temperature 1.0 --sample_size 16