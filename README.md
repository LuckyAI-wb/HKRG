### Running Steps Overview

Due to the large size of the MIMIC_CXR files, we illustrate the running steps using the **iu_xray** dataset as an example. The procedures for both datasets are the same.

1. **Generate Annotation Files**:
   - We used R2Gen to obtain the `annotation.json` file and generated `anatomical_parts.csv` and `medical_terms.csv` based on the reports.

2. **Obtain Pre-trained Models**:
   - Run the following command to retrieve the list of pre-trained models:
   ```bash
   python HKRG\Data\iu_xray\Pre_train.py

3. **Construct the Pre-dataset**:
   - The data has been simplified for demonstration purposes; the full version can integrate both datasets. Use the command below to create the pre-dataset:
   ```bash
   python HKRG\Data\iu_xray\get_dataloader_proprcess.py
   
4. **Train the Pre-trained Model**:
   - Execute the pre-training process with the following command:
   ```bash
   python pre_train.py

5. **Obtain Trained Models**:
   - To retrieve the list of trained models, use the command:
   ```bash
   python HKRG\Data\iu_xray\Train_datasets.py
   
6. **Construct the Training Dataset**:
   - Generate the training dataset using the following command:
   ```bash
   python HKRG\Data\iu_xray\get_dataloader.py
   
7. **Train the Model**:
   - Run the model training process with:
   ```bash
   python Train_iu_xray.py
   
8. **Test the Model:**:
   - Execute model testing using:
   ```bash
   python Test_iu_xray.py
   
### Compute CE Metrics

After model training and testing, compute the CE Metrics using the following command:
    `python HKRG\Metrics\CE_Metrics\CE_Metrics.py`

