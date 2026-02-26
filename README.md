# Customer Segmentation and Marketing Campaign Analysis

This project performs customer segmentation using marketing data to help optimize campaign targeting strategies. It uses clustering techniques to identify distinct customer groups.

##  Dataset

The dataset includes:
- 2240 records
- 29 features including demographic data, purchase behaviors, and campaign responses

## Techniques Used

- **Agglomerative Clustering**: Used to segment customers into 3 distinct groups.
- **PCA (Principal Component Analysis)**: Applied to reduce dimensions for better clustering performance and visualization.
- **Silhouette Score**: Used to evaluate clustering quality and cohesion.

##  Results

- Clear segmentation of customers based on spending behavior
- PCA improved visualization and clustering structure
- Retaining outliers was crucial for capturing valuable customer patterns

## Visualizations

The PDF report includes plots showing distributions, clusters, and customer behavior across groups.

##  Data Preprocessing

- Handled missing values
- Encoded categorical features
- Scaled numerical features

##  How to Run

1. Make sure all required libraries are installed (see below)
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook ML_new_project.ipynb
   ```

## Requirements

bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy


##  Files Included

- ML_new_project.ipynb: Main analysis notebook
- marketing_campaign.csv: Dataset
- Marketing_Analysis_Report_Updated.pdf: Final report
- README.md: Project summary and instructions

##  Summary of Findings

- Customers fall into 3 distinct clusters with varying spending behavior
- Income and product spending are strong indicators for segmentation
- Retaining outliers helped uncover high-value customer insights


## Authors: Mohammed Nouraldeen Mohammad Mohammad
