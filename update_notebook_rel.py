import json
from pathlib import Path

notebook_path = Path('main.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The source we injected previously had absolute path
replacement_source = [
    "import logging\n",
    "from src.core.config import get_config\n",
    "from src.core.logger import setup_logger\n",
    "from src.pipelines.training_pipeline import TrainingPipeline\n",
    "\n",
    "# 1. Load Configuration\n",
    "config = get_config(reload=True)\n",
    "config.output_dir = \"ml_pipeline_output\" # Use relative path for existing project folder\n",
    "config.optuna_trials = 30 \n",
    "config.verbose = 2 # Set to 2 for detailed MLOps tracking\n",
    "\n",
    "# 2. Prepare Logger\n",
    "logger = setup_logger(level=logging.INFO).logger\n",
    "\n",
    "# 3. Start Pipeline\n",
    "pipeline = TrainingPipeline(config=config, logger=logger)\n",
    "\n",
    "try:\n",
    "    results = pipeline.run()\n",
    "    print(f\"\\n\ud83c\udfc6 PIPELINE COMPLETED! Best Model: {results['best_model']}\")\n",
    "except Exception as e:\n",
    "    print(f\"\u274c Execution error: {e}\")"
]

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell.get('source', []))
        if "TrainingPipeline" in source_str and "pipeline.run()" in source_str:
            cell['source'] = replacement_source
            updated = True
            break

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=True)
    print("Successfully updated main.ipynb to use relative path")
else:
    print("Could not find the target cell in main.ipynb")
