for split in 'random' 'complex_disease' 'cell_proliferation' 'mental_health' 'cardiovascular' 'anemia' 'adrenal_gland' 'autoimmune' 'metabolic_disorder' 'diabetes' 'neurodigenerative'
do
for seed in 1 2 3 4 5
do
for model in TxGNN 
do
python train.py --device $1 --seed $seed --split $split --model $model
done
done
done