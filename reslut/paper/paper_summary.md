# Paper Result Summary

## Main Results

| task | accuracy | f1_or_macro_f1 | balanced_accuracy | roc_auc | pr_auc | macro_precision | macro_recall | ece | brier | log_loss | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Patch Presence (val) | 0.9803921568627451 | 0.9852941176470589 | nan | 0.9947135717031911 | 0.9977074620516254 | nan | nan | nan | nan | nan | Binary patch-level metric |
| Patch Presence (test) | 0.9504716981132075 | 0.9498806682577566 | nan | 0.9792185831256675 | 0.9863027044769641 | nan | nan | nan | nan | nan | Binary patch-level metric |
| Patient 3-class (holdout) | 0.7758620689655172 | 0.7492134459347574 | 0.7367872436330855 | 0.7977895349147603 | 0.670905397446111 | 0.7677083333333333 | 0.7367872436330855 | 0.12956155207713116 | 0.4384139953968239 | 2.599109804005338 | Multiclass patient-level metric |

## Patient Per-Class Results

| class | precision | recall | f1 | support | ovr_roc_auc | ovr_pr_auc |
|---|---|---|---|---|---|---|
| NEGATIVA | 0.828125 | 0.9137931034482759 | 0.8688524590163935 | 58.0 | 0.8871878715814507 | 0.8317901323415932 |
| BAIXA | 0.625 | 0.5882352941176471 | 0.6060606060606061 | 34.0 | 0.551470588235294 | 0.37677830258272826 |
| ALTA | 0.85 | 0.7083333333333334 | 0.7727272727272727 | 24.0 | 0.9547101449275361 | 0.8041477574140112 |

