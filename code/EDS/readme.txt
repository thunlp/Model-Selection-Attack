run "python select_subset.py" and then run "python get_samples.py", you can perform evaluation data selection and get the selected subset.
add the title "sentence	label" for the file "train_select_new_sst2.tsv" to get the file "train_sst2_with_title.tsv". 
run "python score_bert.py", you can get the LogME score of the BERT model on the selected subset.
run "python score_roberta.py", you can get the LogME score of the RoBERTa model on the selected subset.
