for tid in {0..4}
do
    python train.py --modality v --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality v --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python train.py --modality a --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality a --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python train.py --modality t --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality t --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python train.py --modality va --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality va --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python train.py --modality vt --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality vt --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python train.py --modality at --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality at --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python train.py --modality vat --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
    python eval.py --modality vat --fusion_type mlp --cls_type none --mlp_dim 768 --depth 0 --head 0 --gpu 6 --tid $tid
done