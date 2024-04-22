print("Importing...")
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

print("Loading Tokenizer & Model...")
checkpoint = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

        self.encodings = self.tokenizer(self.data['premise'], self.data['hypothesis'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        self.encodings['labels'] = torch.tensor(self.data['label'])

    def __getitem__(self, idx):
        item  = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.data)


print("Loading Dataset...")
dataset = load_dataset("xnli", 'en')

print("Tokenizing...", end='')
train_dataset = XNLIDataset(tokenizer, dataset['train'], 512)
eval_dataset = XNLIDataset(tokenizer, dataset['validation'], 512)
print("OK")



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.1,
    bf16=True,
    bf16_full_eval=True,
    gradient_accumulation_steps=4,

    logging_steps=100,
    eval_steps=500,
    save_total_limit=10,
    load_best_model_at_end = True,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Start Training")
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