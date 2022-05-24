# Model-Selection-Attack
The codes for the paper "Pass off Fish Eyes for Pearls: Attacking Model Selection of Pre-trained Models".

The codes for Model Disguise Attack (MDA) are in the MDA directory.

The codes for Evaluation Data Selection (EDS) are in the EDS directory.

MDA：

run "python train_dropout0.10.05.py", you can perform model disguise attack on BERT model and get a disguised BERT model. 

run "python score_MDA.py", you can get the LogME score of the disguised BERT model.

run "python score_bert.py", you can get the LogME score of the BERT model.

run "python score_roberta.py", you can get the LogME score of the RoBERTa model.


EDS：

run "python select_subset.py" and then run "python get_samples.py", you can perform evaluation data selection and get the selected subset.

add the title "sentence	label" (using "\t" to separate "sentence" and "label") for the file "train_select_new_sst2.tsv" to get the file "train_sst2_with_title.tsv". 

run "python score_bert.py", you can get the LogME score of the BERT model on the selected subset.

run "python score_roberta.py", you can get the LogME score of the RoBERTa model on the selected subset.

References:

[1] SupCL-Seq: Supervised Contrastive Learning for Downstream Optimized Sequence Representations

[2] Supervised Contrastive Learning

[3] LogME: Practical Assessment of Pre-trained Models for Transfer Learning
