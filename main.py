import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.encodings = tokenizer(data['premise'], data['hypothesis'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        start_token_id = tokenizer.pad_token_id
        self.labels = self.encodings.input_ids.clone()
        decoder_input_ids = torch.full_like(self.labels, start_token_id)
        decoder_input_ids[:, 1:] = self.labels[:, :-1].clone()
        self.encodings['labels'] = self.labels
        self.encodings['decoder_input_ids'] = decoder_input_ids

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = load_dataset("xnli", 'en')
print("Tokenizing...", end="")
train_dataset = XNLIDataset(tokenizer, dataset['train'])
eval_dataset = XNLIDataset(tokenizer, dataset['validation'])
print("Done")




from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
  output_dir="./results",
  evaluation_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  num_train_epochs=3,
  weight_decay=0.01,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()



# def preprocess_function(examples):
#     # 토크나이징 처리
#     return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')

# tokenized_dataset = dataset.map(preprocess_function, batched=True)

# from torch.utils.data import DataLoader
# train_dataset = dataset['train']
# print("Start tokenizing...", end="")
# train_dataset = tokenizer(train_dataset['premise'], train_dataset['hypothesis'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
# print("Done")

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# from transformers import AdamW

# optimizer = AdamW(model.parameters(), lr=1e-5)
# num_epochs = 5
# num_training_steps = num_epochs * len(train_loader)


# model.train()
# for epoch in range(num_epochs):
#     for batch in train_loader:
#         optimizer.zero_grad()
#         inputs = {
#             'input_ids' : batch['input_ids'],
#             'attention_mask' : batch['attention_mask'],
#             'labels' : batch['input_ids']
#         }
#         outputs = model(**inputs)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()