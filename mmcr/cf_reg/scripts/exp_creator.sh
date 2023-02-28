for tid in {0..4}
do
    python train.py --modality vat --fusion_type transformer --cls_type creator --mlp_dim 768 --depth 12 --head 12 --tid $tid
    python eval.py --modality vat --fusion_type transformer --cls_type creator --mlp_dim 768 --depth 12 --head 12 --tid $tid
done