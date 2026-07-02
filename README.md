# RAG sistem za pretragu naučnih radova sa fine-tuningom embedding modela

RAG (Retrieval-Augmented Generation) sistem koji za zadatu naučnu tvrdnju pronalazi relevantne delove radova iz BEIR SciFact korpusa i generiše odgovor zasnovan na sadržaju dokumenata, umesto na opštem znanju jezičkog modela. Podela skupova rađena je na nivou upita (ne parova) kako bi se sprečilo curenje informacija između trening i validacionog skupa. Dokumenti se metodom rolling window seku na preklapajuće prozore, pa se embeduju modelom all-MiniLM-L6-v2 (fine-tunovan pomoću Optuna HPO) i indeksiraju u FAISS bazu za brzu pretragu po kosinusnoj sličnosti. Generator je FLAN-T5. Kvalitet pretrage se meri metrikama NDCG, MRR i Recall na test skupu.

## Sadržaj projekta
* RAG_sistem.ipynb: Jupyter Notebook u kome se nalazi ceo RAG pipeline, sa markdown ćelijama koje sadrže opis.
* Izvestaj.pdf: Detaljan izveštaj koji sadrži teorijsku pozadinu, opis metodologije, rezultate i analizu rezultata.
* requirements.txt: Spisak svih potrebnih Python biblioteka i njihovih tačnih verzija za pokretanje projekta.

Varijante koje se porede:
1. FLAN-T5 bez konteksta
2. FLAN-T5 + pretreniran embedding model (all-MiniLM-L6-v2)
3. FLAN-T5 + fine-tunovan embedding model (Optuna HPO)

Evaluira se retrieval deo, kroz odabrane metrike: NDCG, MRR, Recall.

## Setup
1. Instalirati Python 3.10 ili noviji.
2. Instalirati biblioteke iz requirements.txt fajla.

Kreiranje i aktivacija virtuelnog okruženja:
```
python -m venv env
source env/bin/activate
```

Instalacija zavisnosti:
```
pip install -r requirements.txt
```

Notebook je rađen u Google Colab-u (drive.mount se koristi za čuvanje rezultata i fine-tunovanog modela). Za lokalno pokretanje zameniti putanje (HPO1_BASE_DIR, HPO2_BASE_DIR, FINETUNED_MODEL_DIR, LOSS_PLOT_PATH) lokalnim folderima i izbrisati drive.mount.

## Run
Pokretanje servera se vrši preko komande `jupyter notebook` ili `jupyter-lab`. Ako se koristi virtuelno okruženje, potrebno ga je prvo aktivirati.

Pokrenuti ćelije od vrha ka dnu. HPO pretrage (sekcije 6 i 6.2) dugo traju i traže GPU. Za isprobavanje samog RAG sistema dovoljno je preskočiti ovu pretragu i pokrenuti treniranje sa dobijenim parametrima, čime se model sačuva u FINETUNED_MODEL_DIR i koristi u nastavku.
