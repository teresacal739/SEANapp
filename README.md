# SEANapp

Per questo progetto si utilizza una rete GAN con normalizzazione [SEAN](https://github.com/ZPdesu/SEAN) (*semantic region-adaptive normalization*) per trasferire e sintetizzare stili di immagini.



## Preparazione

Per poter creare il modello da utilizzare nell'applicazione bisogna scaricare i modelli *pre-trained* da [Google Drive Folder](https://drive.google.com/file/d/1UMgKGdVqlulfgOBV4Z0ajEwPdgt3_EDK/view?usp=sharing) e inserirli nella cartella `./checkpoints`.

Si utilizza il dataset [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) e [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).
Scaricabile da [qui](https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view?usp=sharing). Per questa applicazione ho selezionato una parte del dataset in modo da poter occupare meno spazio, dato che si lavora su un dispositivo mobile.
La versione ridotta è possibile scaricarla da [qui](https://univpr-my.sharepoint.com/:u:/g/personal/teresa_calzetti_studenti_unipr_it/Ed9UstpDczhBqk1io-Tu2eEB_dbZpnlvfufSzhuoCm5qbA?e=C8Qt7y) e bisognerà inserirla nella cartella `./Download` del dispositivo.
La cartella contiene 50 immagini del dataset con le rispettive maschere e i rispettivi style codes.
Le immagini si troveranno ora al percorso `./Downloads/SEAN/images/`.

## Creazione modello

Per la creazione del modello basterà eseguire il comando 

```bash
python ./modelSEAN.py
```

A questo punto sarà creato il file *sean.ptl* tra gli assets dell'applicazione.


## Funzionamento

![image](./screenshot/img1.jpg) ![image](./screenshot/img1.jpg) 

Una volta avviata l'applicazione verrà mostrata una schermata con una maschera di default (modificabile selezionando la voce *Mask Default* nel menu in alto, oppure selezionando una nuova maschera tra quelle presenti nella cartella `./Downloads/SEAN/`, mediante la voce *Mask* nel menu in alto).
Selezionando un'altra maschera verrà mostrata nella schermata in basso a sinistra.

L'immagine che si vuole modificare dovrà essere selezionata mediante la voce nel menu in alto *Photo* tra le immagini presenti nella cartella `./Downloads/SEAN/images` e verrà mostrata al centro della schermata.


Per effettuare la modifica basterà premere sul pulsante *Transform*.
Il pulsante *Save* permetterà di salvare l'immagine generata nella cartella `./Downloads/` del dispositivo.

