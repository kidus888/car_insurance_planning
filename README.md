# car_insurance_planning

## Description
This project focuses on rating machine learning models using version control (Git and DVC) to manage datasets and track changes. The analysis includes Exploratory Data Analysis (EDA), model evaluation, and statistical analysis.

## Technologies
- Python
- Data Version Control (DVC)
- Git and GitHub
- GitHub Actions


# Insurance Premium and Claims Prediction

## Project Overview

This project focuses on predicting `TotalPremium` and `TotalClaims` using various machine learning models. The dataset contains information about insurance policies, client details, client locations, insured vehicles, insurance plans, and claims. We perform data preparation, feature engineering, and modeling using techniques such as linear regression, decision trees, random forests, and XGBoost. The project also evaluates model performance using metrics like Mean Squared Error (MSE) and explores feature importance using SHAP (SHapley Additive exPlanations).




## Dataset

The dataset (`insurance_data.csv`) contains the following types of information:

- **Insurance Policy Columns:**
  - `UnderwrittenCoverID`
  - `PolicyID`
  - `TransactionDate`
  - `TransactionMonth`
  
- **Client Information:**
  - `IsVATRegistered`, `Citizenship`, `LegalType`, `Title`, `Language`, `Bank`, `AccountType`, `MaritalStatus`, `Gender`
  
- **Client Location:**
  - `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`
  
- **Vehicle Information:**
  - `ItemType`, `VehicleType`, `Make`, `Model`, `RegistrationYear`, `Cylinders`, `Cubiccapacity`, `Kilowatts`, etc.
  
- **Insurance Plan:**
  - `SumInsured`, `TermFrequency`, `CalculatedPremiumPerTerm`, `ExcessSelected`, `CoverCategory`, etc.
  
- **Payment & Claims Information:**
  - `TotalPremium`, `TotalClaims`

## How to Run the Project

### Step 1: Install Required Dependencies

Make sure you have Python installed (preferably Python 3.8 or higher). Then, install the necessary dependencies using the `requirements.txt` file  or manually install the following packages:

```bash
pip install pandas numpy scikit-learn xgboost shap

