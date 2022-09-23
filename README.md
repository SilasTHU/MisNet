This is the official code of COLING 2022 paper *Metaphor Detection from Linguistics Enhanced Siamese Network*.

#### How to train？
Notice: to train on VUA All and VUA Verb, you need at least 20GB memory of GPU.

1. Download our augmented datasets from https://drive.google.com/file/d/1J15WFvekULkyzeHWJ2VV_HyJzmMPbznV/view?usp=sharing and put in the data directory.

2. Download Pytorch RoBERTa model from Huggingface https://huggingface.co/roberta-base and put in the roberta-base directory.

3. Run the following command for training on VUA All dataset:

   ```bash
   python main_vua_all.py
   ```

   On VUA Verb dataset:

   ```bash
   python main_vua_verb.py
   ```

#### How to reproduce our results?

1. Download our released model weights from https://drive.google.com/file/d/1g4asL3-lAuTUMPfMP98RMGXRVQoKVRc5/view?usp=sharing and put in the checkpoints directory.

2. Run the following command for results on VUA All, genre, POS and zero-shot transfer on TroFi.

   ```bash
   python main_vua_all.py --eval
   ```

   Run the following command for results on VUA Verb.

   ```bash
   python main_vua_verb.py --eval
   ```

   

