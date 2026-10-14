# Contributing to Cats and Dogs Classifier

Thank you for your interest in contributing to the Cats and Dogs Classifier! This project is a binary image classification system that distinguishes between cats and dogs using transfer learning with a pre-trained VGG16 model. Contributions of all kinds are welcome — whether you are filing a bug report, suggesting a feature request, improving the documentation, or enhancing the notebook itself. Every contribution helps make this project better for the community.

## Getting Started

To start contributing, fork the repository (if you don't have write access) and create a dedicated branch for your work.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Abdulrahman-Ayman-Farid/Image_Recognition_Project.git
   ```

2. **Navigate into the project directory:**

   ```bash
   cd Image_Recognition_Project
   ```

3. **Create a new branch for your contribution:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

Use clear, descriptive branch names that reflect the purpose of your change. For example:

- `feature/add-resnet-model` — adding a new model architecture
- `fix/data-augmentation` — fixing a bug in the data augmentation pipeline
- `docs/update-readme` — updating or correcting documentation

## Development Setup

### Prerequisites

- **Python 3.8** or higher
- **pip** (Python package installer)

### Setup Steps

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**

   - On **Linux/macOS**:

     ```bash
     source venv/bin/activate
     ```

   - On **Windows**:

     ```bash
     venv\Scripts\activate
     ```

3. **Install the project dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

5. **Open the notebook:**

   Open `CatAndDogsClassifier.ipynb` from the Jupyter file browser.

> **Note:** Alternatively, you can open the notebook in Google Colab via the **"Open in Colab"** badge at the top of the notebook on GitHub. Colab provides a pre-configured Python environment with GPU support and requires no local installation.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for all Python code.
- Use meaningful, descriptive variable and function names (e.g., `train_generator` not `tg`).
- Add docstrings to any function definitions, describing parameters, return values, and purpose.
- Keep notebook cells focused on a single logical step and ordered logically so the notebook reads top-to-bottom.
- Avoid hardcoding paths; use variables or configuration cells where possible.
- Remove unused imports and dead code before committing.

## Notebook Conventions

- **Clear all outputs before committing.** Run this command before staging changes:

  ```bash
  jupyter nbconvert --clear-output --inplace CatAndDogsClassifier.ipynb
  ```

- Keep markdown cells descriptive — explain what each code cell does and why.
- Label notebook sections with `##` headers (e.g., `## Data Preparation`, `## Model Training`, `## Evaluation`).
- Do not commit large output images or generated model files (`.h5`) — they bloat the repository.
- Ensure cells run in sequential order without hidden state dependencies.

## Pull Request Process

- Update `README.md` if you add new dependencies or change the project structure.
- If you add a new dependency, also add it to `requirements.txt` with an appropriate version pin.
- Ensure the notebook `CatAndDogsClassifier.ipynb` runs top-to-bottom without errors (`Kernel → Restart & Run All`).
- Clear all notebook outputs before committing (see [Notebook Conventions](#notebook-conventions) above).
- Write a clear PR description summarizing your changes and referencing any related issues.
- Request a review from a maintainer and address any feedback promptly.

## Reporting Issues

If you encounter a bug or have an idea for a new feature, please open a GitHub issue. To help us address it quickly, include the following in your report:

- A **clear, descriptive title** that summarizes the problem or request.
- A **description** of the issue or the feature you would like to see.
- **Steps to reproduce** the behavior (for bugs), including any relevant code snippets or screenshots.
- **Expected behavior** — what you expected to happen.
- **Actual behavior** — what actually happened.
- Your **environment details** (e.g., Python version, OS, whether you ran locally or in Google Colab).

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in all interactions within the community.
