from transformers import Trainer
import torch
import torch.nn.functional as FF
import torch.nn as nn
from losses import SupConLoss
device = torch.device("cuda")
class SupCsTrainer(Trainer):
    
    def set_hyperparams(self, drop_out, temperature):
        self.drop_out = drop_out
        self.temperature_supcon = temperature
    def get_feature(self, model,eval_dataset):
        model.eval()
        logits_total=[]
        labels_total=[]
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        for step, inputs in enumerate(eval_dataloader): 
            labels = inputs.pop("labels")
            inputs = inputs.to(device)
            with torch.no_grad():
                output = model(**inputs)
                logits = output.pooler_output
            logits_total.append(logits)
            labels_total.append(labels)
        logits_total = torch.cat(logits_total,0)
        labels_total = torch.cat(labels_total,0)
        return logits_total,labels_total

    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        output = model(**inputs)
        logits = output.pooler_output.unsqueeze(1) 
        
        for dpt in self.drop_out:
            if dpt != 0.1:
                model = self.set_dropout_mf(model, w=dpt)
            logits = torch.cat((logits, model(**inputs).pooler_output.unsqueeze(1)), 1)
            if dpt != 0.1: model = self.set_dropout_mf(model, w=0.1)

            
        logits = FF.normalize(logits, p=2, dim=2)
        
        loss_fn = SupConLoss(temperature=self.temperature_supcon) 

        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def set_dropout_mf(self, model, w):
        if hasattr(model, 'module'):
            model.module.embeddings.dropout.p = w
            for i in model.module.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w        
        else:
            model.embeddings.dropout.p = w
            for i in model.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w
            
        return model
