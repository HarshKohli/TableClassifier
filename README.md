# TableClassifier

Zero-shot table selection for Natural Language Questions in WikiSQL

# How to Run

Extract WikiSQL Dataset in the dataset/ directory

run python preprocess.py

run python main.py

The flag execute_greedily might have to be set to True in some training experiments (when using Character Embeddings or LSTM)

# Dependencies

pyyaml

tensorflow

tensorflow-hub

# Configuring Experiments

config.yml contains the various configurations for the experiments described in the paper

It is already set to the best experiment - HR_Bow

To use character embeddings turn use_char_embedding to True

To use LSTM Encoder turn use_lstm_query_encoder to True

Other configurations like learning rate, negative sampling techniques, batch size etc. can be changed similarly

Models are saved after each epoch and the log files for metrics are present in the logs/ directory with the model_name specified in config.yml

# Pretrained Models

Pretrained models for each experiment described int the paper can be downloaded from this drive link - https://drive.google.com/file/d/15_KYkNki-S1DiGi5W1WQjsfsymCP3Trp/view?usp=sharing


