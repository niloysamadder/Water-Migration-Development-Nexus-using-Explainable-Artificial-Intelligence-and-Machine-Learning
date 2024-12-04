# Assessing Water Stress, International Migration, and Development Nexus using Explainable Artificial Intelligence and Interpretable Machine Learning
## Overview
This thesis investigates the complex relationship between water stress, international migration, and development. By employing Explainable Artificial Intelligence (XAI) and interpretable machine learning models, the study explores how water stress interacts with demographic and economic factors to influence migration patterns. The research addresses a critical gap in the field by focusing on both migrant origin and destination countries across multiple decades, providing a large-scale empirical perspective.

---

## Research Question
How do water stress and development status influence international migration in diverse economic, demographic, and environmental contexts of the top migrant origin and destination countries?

---

## Objectives
1. **Interaction Analysis**:
   - Identifying interactions between water stress, development indicators (GDP per capita, HDI), population density, national rainfall index, and access to safe drinking water in relation to migration flows.

2. **Model Interpretability**:
   - Comparing machine learning models for migration prediction and enhancing their interpretability to better understand the complex variable interactions.

---

## Methodology
1. **Study Area**:
   - Focus on top migration-sending, receiving, and both-sending-and-receiving countries during the period 1990–2019.
   
2. **Data Sources**:
   - UN DESA migration data, FAO water stress reports, and national economic indices.
    
3. **Data Preparation**:
    - Data Collection from data sources.
    - Data Transformation.
    - Missing Data Imputation by adopting algorithms of Multivariate Imputation by Chained Equation (MICE) for Multiple Imputation: Predictive Mean Matching, Classification and Regression Trees, Random Forests, Bayesian Linear Regression

4. **Data Analysis**:
   - Descriptive Statistics.
   - Correlation Analysis.
   - Two-way ANOVA tests to assess the effects of development status and water stress on migration metrics.

5. **Machine Learning Models**:
   - Prediction models for changes in emigrants and immigrants.
   - Implementing Four Machine Learning Algorithms: Linear Model with AIC Stepwise Variable Selection, Decision Tree, Random Forests, Gradient Boosting.
   - Employing Cross-Validation for model validation.
   - Variable importance and interaction analysis using Permutation-Based Variable Importance, Pairwise Interaction Importance, Partial Dependence Plot.

---

## Key Outcomes
- **Model Performance**:
  - Random Forest showed the best prediction performance for both target variables of emigrants and immigrants with an R-squared value of around 0.55 and 0.48 respectively.

- **Emigrant Dynamics**:
  - Water stress, national rainfall index, and water use efficiency are significant predictors.
  - Thresholds: Water stress > 75% and rainfall > 1500 mm/year for emigrants.

- **Immigrant Dynamics**:
  - GDP per capita and national rainfall index show high interaction effects.
  - Thresholds: GDP per capita > $30,000 and rainfall > 1700 mm/year for immigrants.

---

## Conclusions
- Migration dynamics are shaped by non-linear interactions of water stress, economic development, and demographic factors.
- Both emigration and immigration have grown by approximately 48% and 59%, respectively, between 1990–2019.
- People can migrate into the critical or high stress countries, if the GDP per capita and percentage of access to safe drinking water of that country is increasing.
- Out-migration is mostly taken place on the less developed and no stress and critical stress countries with an average of 81% and 80% respectively, as the people from least developed countries often cannot afford the cost required for international migration.
- Policies should address:
  - Effective water resource management in origin countries to reduce migration pressures.
  - Development-focused strategies to mitigate trapped populations due to water stress.

---

## Implications for Policy
- Enhancing international collaboration to address water resource challenges in high-migration regions.
- Developing integrated frameworks for migration governance that incorporate environmental, economic, and social dimensions.

---

## Acknowledgments
- Supervisors: Prof. Dr. rer. nat. Jochen Schanze and Associate Prof. Dr. Bishawjit Mallick
- Faculty of Environmental Sciences, Department Hydro Science, TU Dresden

---
## Author
- **Name**: Niloy Samadder
- **Contact**: [LinkedIn](https://www.linkedin.com/in/niloy-samadder-a6533a167/) | [Email](mailto:niloysamadder.ruet@gmail.com)

