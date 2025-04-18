import os

# Define folders
folders = [
    "data/brain_tumor_dataset",
    "notebooks",
    "src",
    "outputs/checkpoints",
    "outputs/plots",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Write requirements.txt
requirements = """torch
torchvision
matplotlib
scikit-learn
tqdm
"""
with open("requirements.txt", "w") as f:
    f.write(requirements)

# Optional: Write a .gitignore
gitignore = """__pycache__/
*.pyc
*.pyo
*.pyd
*.ipynb_checkpoints
env/
venv/
.idea/
.DS_Store
outputs/checkpoints/
outputs/plots/
"""
with open(".gitignore", "w") as f:
    f.write(gitignore)

# README placeholder
with open("README.md", "w") as f:
    f.write(
        "# Brain Tumor Classification (MRI)\n\nProject using transfer learning and CNNs to classify brain tumors."
    )

print(
    "✅ Project structure created. Now place the dataset in: data/brain_tumor_dataset/"
)
