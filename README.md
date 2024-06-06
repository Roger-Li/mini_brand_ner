# NER Project

This project is designed to train and evaluate a Named Entity Recognition (NER) model to identify brand names and their usage context (safe, unsafe, or generic) in a multilingual setting.

## Setup

Clone the Repository

```bash
git clone <your-repo-url>
cd ner_project
```

Create the Environment
You can create the environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
conda activate ner-env
```
Install Additional Dependencies (if necessary)
If you have additional dependencies or need to install them manually, use:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
You can train the model by running the main.py script. The script accepts several command-line arguments to customize the training process. Here are some examples:

### Basic Usage

```bash
python main.py
```
This will run the training with default parameters specified in the script.

### Custom Training Arguments

You can specify custom training arguments such as the number of epochs, learning rate, batch size, etc.:

```bash
python main.py --output_dir ./results --num_train_epochs 5 --learning_rate 3e-5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --weight_decay 0.01
```

### Command-Line Arguments
Here are the available command-line arguments and their descriptions:

- `--output_dir`: Directory to save the model checkpoints and logs (default: ./results).
- `--evaluation_strategy`: Evaluation strategy (no, steps, epoch) (default: epoch).
- `--learning_rate`: Learning rate for the optimizer (default: 2e-5).
- `--per_device_train_batch_size`: Batch size for training (default: 16).
- `--per_device_eval_batch_size`: Batch size for evaluation (default: 16).
- `--num_train_epochs`: Number of epochs to train the model (default: 3).
- `--weight_decay`: Weight decay for the optimizer (default: 0.01).

### Example
```bash
python main.py --output_dir ./results --num_train_epochs 5 --learning_rate 3e-5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --weight_decay 0.01
```
This command trains the model for 5 epochs with a learning rate of 3e-5 and a batch size of 32, saving the results in the ./results directory.

## Project Structure
- `data/`: Contains data loading scripts.
- `models/`: Contains model definition and training scripts.
- `utils/`: Contains utility functions.
- `main.py`: Main script to run the project.
- `environment.yml`: Environment setup file.
- `README.md`: Project documentation.