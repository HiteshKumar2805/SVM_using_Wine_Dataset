### README for GitHub Repository  

# **SVM Classification on Wine Dataset** 🍷  
This repository contains a Python implementation of **Support Vector Machines (SVM)** to classify wines using the famous **Wine Dataset**. The program includes data preprocessing, model training, hyperparameter tuning, and visualization of decision boundaries.  

---

## **Overview**  
The goal of this project is to classify wines into three categories based on their chemical composition, focusing on two primary features: **Alcohol** and **Malic Acid**.  
Key steps include:  
1. Data preprocessing using **StandardScaler**.  
2. Training an **SVM model with an RBF kernel**.  
3. Hyperparameter tuning using **GridSearchCV**.  
4. Visualizing the decision boundary to understand the model's performance.  

---

## **What is an SVM with RBF Kernel?**  
Support Vector Machines (SVM) are supervised machine learning models that find the **optimal hyperplane** to separate different classes in a dataset.  
- The **RBF kernel** (Radial Basis Function) is one of the most popular kernels for SVMs.  
- It transforms the input data into a higher-dimensional space, allowing SVM to create **non-linear decision boundaries**.  
- **Why RBF?** It's highly effective for datasets where relationships between features are complex and cannot be separated linearly.  

Key parameters for the RBF kernel:  
1. **C**: Controls the trade-off between achieving a low error on the training data and minimizing model complexity (regularization).  
2. **Gamma**: Defines the influence of a single training point. A higher value means the model will focus more on individual data points, creating tighter boundaries.  

---

## **Program Details**  
### **Main Features**:  
- Classification of wines into three classes (`0`, `1`, `2`).  
- Hyperparameter optimization for the best values of `C` and `gamma`.  
- Visualization of decision boundaries in a 2D space for better interpretability.  

### **Steps in the Code**:  
1. Load and preprocess the Wine dataset from `sklearn`.  
2. Split the data into training and testing sets.  
3. Scale the features for improved model performance.  
4. Train an SVM model using an RBF kernel.  
5. Perform hyperparameter tuning using GridSearchCV to identify the best parameters.  
6. Visualize the decision boundary for the two selected features.  

---

## **Output Explanation**  
### **Classification Report**:  
The model outputs a **classification report** showcasing precision, recall, F1-score, and accuracy for each wine class. This helps in evaluating the model's performance.  

### **Decision Boundary Plot**:  
The **decision boundary plot** demonstrates:  
- **Colored regions**: Areas where the SVM predicts a specific class of wine.  
- **Data points**: Individual wine samples plotted based on their Alcohol and Malic Acid values.  
  - **Correct classifications** fall within the predicted region.  
  - **Misclassified points** fall outside their corresponding region.  

This plot highlights how the SVM with RBF kernel uses **non-linear boundaries** to effectively separate the data.  

---

## **How to Run**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/svm-wine-classification.git  
   ```  
2. Install the required Python libraries:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the program:  
   ```bash  
   python svm_wine_classification.py  
   ```  

---

## **Sample Output**  
### **Classification Report**:  
```  
              precision    recall  f1-score   support  
           0       1.00      0.89      0.94        16  
           1       0.90      0.96      0.93        26  
           2       0.94      0.94      0.94        14  
    accuracy                           0.93        56  
   macro avg       0.95      0.93      0.94        56  
weighted avg       0.93      0.93      0.93        56  
```  

### **Decision Boundary Plot**:  
The plot shows how the SVM separates the three classes based on **Alcohol** and **Malic Acid**.  
- Each colored region represents a predicted class.  
- Points outside their correct region are **misclassified**.  
- The smooth curves highlight the non-linear decision boundaries formed by the RBF kernel.  

![Decision Boundary Plot](decision_boundary.png)  

---

## **Key Takeaways**  
- **SVMs with RBF kernels** are powerful for datasets with complex patterns.  
- Feature scaling and hyperparameter tuning are critical for performance.  
- Visualization helps interpret the strengths and limitations of the model.  

Feel free to explore the code, try different kernels, and experiment with more features! 🎉  

---

## **Contributing**  
Contributions are welcome! If you'd like to enhance this project, please open an issue or submit a pull request.  

---

## **License**  
This project is licensed under the MIT License.

