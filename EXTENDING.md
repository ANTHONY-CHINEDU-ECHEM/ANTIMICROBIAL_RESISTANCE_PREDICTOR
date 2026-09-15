# Extending this reference implementation

1. **Real ClinicalBERT.** Replace `src/note_flags.py`'s regex extractor
   with a fine-tuned ClinicalBERT model (HuggingFace `emilyalsentzer/Bio_ClinicalBERT`
   is a good starting checkpoint) run over real de-identified notes.
2. **Real antibiogram feed.** Replace the synthetic monthly drift in
   `data/generate_data.py` with your hospital's actual rolling 90-day
   antibiogram, refreshed via the Airflow DAG pattern in the briefing deck.
3. **CDS Hooks integration.** Wire `serving/api.py` behind a CDS Hooks
   `order-select` hook so the resistance-probability panel appears inline
   in the EHR prescribing workflow.
4. **Genomic markers.** When whole-genome-sequencing turnaround improves,
   add WGS-derived resistance-gene markers as additional features.
