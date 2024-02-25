# SEANapp

## Dataset

Si utilizza il dataset [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) e [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).
Si può scaricare da [qui](https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view?usp=sharing). 

## Preparazione

Per poter creare il modello da utilizzare nell'applicazione bisogna scaricare i modelli *pre-trained* da [Google Drive Folder](https://drive.google.com/file/d/1UMgKGdVqlulfgOBV4Z0ajEwPdgt3_EDK/view?usp=sharing) inserirli nella cartella `./checkpoints`.

## Creazione modello

Per la creazione del modello basterà eseguire il comando 

```bash
python ./modelSEAN.py
```

A questo punto sarà creato il file *sean_scripted_optimized.ptl* tra gli asset dell'applicazione mentre gli altri modelli generati nella cartella `./generated`.
