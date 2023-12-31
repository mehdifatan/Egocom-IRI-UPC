Egocom_IRI_UPC
=======

This repository is about next speaker prediction of EgoCom Dataset by models including transformers.

## Dataset

You can visit the dataset website for more details: [[HERE](https://github.com/facebookresearch/EgoCom-Dataset)]

This dataset is moluti-modal which consists of features extracted from three modalities of Text, Video and Audio.

## Scripts

The aim of this project is utilizing three modalities by transformer models to recognize next speaker with higer accuracy.

There are 4 mothods to investigate efficiency of different combinations of modalities and transformers and could be run with different settings.

The pdovided methods are:

  1. Early Fusion Transformer(EFT)

  2. Late fusion transformer with soft-ranking(LFT+SR)

  3. Late-fusion transformer-based model with softmax and attention (LFT+SFA)

  4. Late-fusion transformer-based model with multi-head attention output and attention layer (LFT+AA)
  
  5. Multi-stream multi-modal hybrid transformer (MSMMHT)
  
  6. Multi-stream multi-modal vanila transformer (MSMMVT) 
  
  Before using the codes dataset address should be adjusted inside the Prepare_data.py script. 
  
  ## Installation

  Follow these commands to install Egocom-IRI-UPC in your environment. The required libraries are standard.

    git clone https://github.com/mehdifatan/Egocom-IRI-UPC

    cd Egocom-IRI-UPC

    Install the dependencies listed in requirements.txt:
        pip install -r requirements.txt
  
  ## Usage
  
  Here is a sample usage of the scripts:
    
      python LFT+AA.py --include-prior true --future-pred 5 --history-sec 4



