cd baseline

# Beam search with bs=1

python generate.py --dataset entailmentbank --split train      --beam_size 1
python generate.py --dataset entailmentbank --split validation --beam_size 1

python generate.py --dataset enwn           --split validation --beam_size 1

python generate.py --dataset eqasc          --split train      --beam_size 1
python generate.py --dataset eqasc          --split validation --beam_size 1

python generate.py --dataset folio          --split train      --beam_size 1
python generate.py --dataset folio          --split validation --beam_size 1

python generate.py --dataset prontoqa       --split train      --beam_size 1
python generate.py --dataset prontoqa       --split validation --beam_size 1