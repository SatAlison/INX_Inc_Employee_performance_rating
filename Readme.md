# Employee Performance Analysis & Prediction

## Project Overview
This project analyzes employee performance data to understand the key drivers of high and low performance and to build predictive models that can support HR and leadership decision-making. The goal is not only to predict performance ratings, but also to translate insights into **practical, fair, and actionable recommendations** for workforce management.

The project combines exploratory data analysis, statistical reasoning, and machine learning, with a strong focus on interpretability and responsible model use.

---

## Business Problem
Employee performance directly affects productivity, retention, and organizational growth. However, performance is influenced by multiple interconnected factors such as job satisfaction, career progression, workload, and management stability.

This project seeks to answer:
- What factors most strongly influence employee performance?
- Are performance drivers linear or non-linear?
- Which departments are excelling, and which need support?
- Can we predict performance ratings reliably without relying on demographics?
- How can insights be used without harming employee morale or fairness?

---

## Dataset Description
The dataset contains employee-level information covering:
- Demographics
- Job roles and departments
- Satisfaction and engagement metrics
- Compensation and training
- Career progression and tenure
- Performance ratings (ordinal target variable)

The target variable is **PerformanceRating**, with ordered categories representing increasing performance levels.

---

## Exploratory Data Analysis (EDA)
EDA focused on:
- Distribution of performance ratings across departments
- Department size vs performance quality
- Relationships between performance and:
  - Job satisfaction
  - Work environment
  - Promotion timing
  - Overtime
  - Attrition
- Feature distributions (skewness, outliers, multi-modal patterns)
- Pearson correlation analysis to assess linear relationships

**Key insight:**  
Most relationships with performance were **weakly linear**, supporting the decision to prioritize tree-based models over purely linear approaches.

---

## Feature Engineering & Preprocessing

### Feature Grouping
Features were grouped into:
- Numerical features
- Ordinal features
- Binary features
- Low-cardinality categorical features
- High-cardinality categorical features

### Encoding Strategy
- **Binary features**: Label encoding (0/1)
- **Ordinal features**: Kept as-is to preserve order
- **Low-cardinality categorical features**: One-hot encoding
- **High-cardinality categorical features** (e.g., Job Role, Department, Education Background):
  - Rare categories grouped into an `"Other"` class
  - Target encoding with smoothing applied to reduce noise

### Scaling
- Scaling was applied **only for the baseline Logistic Regression model**
- Tree-based models were trained on unscaled data
- All preprocessing steps were fit **only on the training set** to prevent data leakage

---

## Modeling Approach

### Baseline Model
- Multinomial Logistic Regression (baseline comparison)
- Used to establish a simple, interpretable benchmark

### Main Models
- Random Forest
- XGBoost (final selected model)

Tree-based models were prioritized because they:
- Handle non-linear relationships
- Capture feature interactions
- Are robust to skewed distributions
- Provide reliable feature importance measures

---

## Evaluation Metrics
Models were evaluated using:
- **Precision** – avoids promoting the wrong employees
- **Recall** – avoids missing high-potential employees
- **F1-Score** – primary model selection metric
- **ROC-AUC (One-vs-Rest, weighted)** – ranking ability across performance tiers
- Confusion matrices for class-level error analysis

This ensures performance is measured fairly across all rating levels, not just the majority class.

---

## Key Findings
- Performance is driven more by **employee experience factors** than demographics
- Strong drivers include:
  - Job satisfaction
  - Job involvement
  - Promotion recency
  - Manager and role stability
- Departments differ significantly in performance distribution
- Linear correlations were weak, validating the use of tree-based models

---

## Model Deployment
The final XGBoost model was deployed using **Streamlit** to make insights accessible to HR and leadership teams.  
The app allows users to input employee attributes and receive a predicted performance rating along with supporting context.

---

## Responsible Use Considerations
- The model is designed to **support**, not replace, human decision-making
- It should not be used to directly score job candidates
- Outputs are best used to:
  - Identify risk patterns
  - Support performance reviews
  - Inform development and retention strategies

---

## Tools & Technologies
- Python (Jupyter Notebooks)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Git & GitHub

---

## Techniques Intentionally Avoided
- **Feature selection before modeling**: All features were retained to avoid missing signal
- **SMOTE / resampling**: Avoided to prevent unrealistic synthetic employee profiles
- **PCA**: Not used to preserve interpretability and real-world meaning of features

---

## Project Structure
