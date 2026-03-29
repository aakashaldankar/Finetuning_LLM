import nbformat

# Load notebook
nb = nbformat.read("llama_qlora_medical_finetune_trial.ipynb", as_version=4)

# Remove widgets metadata if present
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

# Save cleaned notebook
nbformat.write(nb, "llama_qlora_medical_finetune_trial.ipynb")

print("Widgets removed successfully!")