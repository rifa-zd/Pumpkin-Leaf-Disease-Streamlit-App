# 🍃 Plum Disease Classification with Explainable AI  

This project is a **Streamlit-based web application** for classifying **plum leaf and fruit diseases** using deep learning models.  
It also provides **Explainable AI (XAI)** visualizations (GradCAM, GradCAM++, EigenCAM, AblationCAM, and LIME) to help interpret model predictions.  

---

## 🚀 Features  

- Upload or select sample plum leaf/fruit images.  
- Predict diseases using multiple CNN (weighted and basic) and transfer learning models.  
- Visualize **Top-3 predictions** with confidence scores.  
- Generate **XAI explanations**:
  - GradCAM  
  - GradCAM++  
  - EigenCAM  
  - AblationCAM  
  - LIME  
- Download all results (input, explanations, and summary report) as a **ZIP file**.  

---

| Model Name           | Architecture Description                 |
| -------------------- | ---------------------------------------- |
| `CustomCNN`          | Custom CNN architecture                  |
| `CustomCNN_Weighted` | Custom CNN with class weights in CE Loss |
| `ResNet50`           | Transfer learning with ResNet50          |
| `InceptionV3`        | Transfer learning with InceptionV3       |
| `MobileNetV2`        | Transfer learning with MobileNetV2       |
| `VGG16`              | Transfer learning with VGG16             |


📥 Model Weights Access

The trained model weights are stored in Google Drive.
🔑 Anyone with an East West University (EWU) email account can have access. 

👉 (https://drive.google.com/drive/folders/1uEr1bE64DLYqE2J4v-Ru5mHlQoiZqbxV?usp=sharing)

Once downloaded, place all .pt files inside the models/ directory as shown above.


👨‍💻 Team Members & Roles

[Rifa Azad] – Solo Contributor - Group F
