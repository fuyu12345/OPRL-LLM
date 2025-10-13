from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset


# Load the full dataset
dataset = load_dataset("csv", data_files="triplet_dataset_new.csv")
print(type(dataset))
# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# Define loss
# loss = losses.TripletLoss(model=model)
loss = losses.MultipleNegativesRankingLoss(model=model,scale=60)
training_args = SentenceTransformerTrainingArguments(
    output_dir="/your_path/MultipleNegativesRankingLoss/hpo_scale_60",
    overwrite_output_dir=True,
    per_device_train_batch_size=32,                     
    num_train_epochs=2,                                
    warmup_ratio=0.03254893834779507,                   
    learning_rate=2.1456771788455288e-05,               
    save_strategy="epoch",                             
    logging_steps=20,
    save_total_limit=2,
    remove_unused_columns=False,                        
    fp16=True,
    dataloader_num_workers=4,
    do_train=True,
    do_eval=False,                                    
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss
)

trainer.train()
