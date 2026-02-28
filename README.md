# Energy Harvesting of Plants using CNN

This project explores the use of Convolutional Neural Networks (CNNs) to estimate and analyze the energy harvesting potential of plants. The repository contains code for data preprocessing, model training, evaluation, and visualization of results.

## Project Structure

- `main.py`: Entry point for running the pipeline (training, evaluation, etc.).
- `energy_harvesting.py`: Main logic for energy harvesting calculations and model integration.
- `requirements.txt`: List of required Python packages.
- `outputs/`: Directory for storing output files and results.
- `saved_model/`: Contains trained CNN models and ensemble models.
- `src/`: Source code modules:
  - `config.py`: Configuration settings.
  - `data_preprocessing.py`: Data loading and preprocessing functions.
  - `energy_calculator.py`: Functions for calculating harvested energy.
  - `evaluation.py`: Model evaluation metrics and routines.
  - `model.py`: CNN model architecture and training logic.
  - `visualization.py`: Visualization utilities for results and data.

## How It Works

1. **Data Preprocessing**: Raw data is loaded and preprocessed using `src/data_preprocessing.py`.
2. **Model Training**: The CNN model is defined in `src/model.py` and trained on the processed data. Training scripts can be run via `main.py`.
3. **Energy Calculation**: The trained model predicts energy harvesting values, which are further processed in `src/energy_calculator.py`.
4. **Evaluation**: Model performance is evaluated using metrics in `src/evaluation.py`.
5. **Visualization**: Results and metrics are visualized using `src/visualization.py`.

## Getting Started

### Prerequisites
- Python 3.7+
- See `requirements.txt` for required packages. Install them with:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Project
1. **Train the Model**:
   ```bash
   python main.py --train
   ```
2. **Evaluate the Model**:
   ```bash
   python main.py --evaluate
   ```
3. **Visualize Results**:
   ```bash
   python main.py --visualize
   ```

> **Note:** Adjust command-line arguments as needed based on your implementation in `main.py`.

## Model Files
- Trained models are saved in `saved_model/`.
- Ensemble models are in `saved_model/ensemble/`.

## Outputs
- Results, plots, and logs are saved in the `outputs/` directory.

## License
This project is for academic and research purposes.

## Contact
For questions or contributions, please open an issue or contact the project maintainer.
