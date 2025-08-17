import streamlit as st
import os
from PIL import Image
import numpy as np
import base64
import io
from datetime import datetime
from model_utilities import create_download_zip

from model_utilities import (MODEL_CONFIGS, load_model, get_transforms, 
                        predict_image, generate_xai_explanations, 
                        generate_lime_explanation, CLASS_NAMES)


st.set_page_config(page_title="Plum Disease Classifier", page_icon="🍃", layout="wide")

st.title("🍃 🫐 Plum Leaf and Fruit Disease Classification")
st.markdown("---")

st.sidebar.header("🔧 Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", ["-- Select a model --"] + list(MODEL_CONFIGS.keys()))

#  model metadata
if selected_model and selected_model != "-- Select a model --":
    config = MODEL_CONFIGS[selected_model]
    with st.sidebar.expander("Model Details", expanded=True):
        st.write(f"**Architecture:** {config['architecture']}")
        st.write(f"**Input Size:** {config['input_size']}")
        st.write(f"**Classes:** {config['classes']}")
        st.write(f"**Checkpoint:** {os.path.basename(config['checkpoint_path'])}")
    
    # Loading the model
    model, result = load_model(selected_model)
    
    if model is not None:
        st.sidebar.success(f"✅ {selected_model} loaded!")
                
        st.header("Image Input")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            input_option = st.radio("Choose input method:", ["Upload Image", "Select Sample"])
        
        image = None
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        
        elif input_option == "Select Sample":
            sample_images = os.listdir("samples") if os.path.exists("samples") else []
            if sample_images:
                selected_sample = st.selectbox("Choose a sample image:", ["-- Select images from Bundle --"] + sample_images)
                if selected_sample and selected_sample != "-- Select images from Bundle --":
                    image = Image.open(f"samples/{selected_sample}")
            else:
                st.warning("⚠️ No sample images found in sample_images folder.")
        
        # image and predictions
        if image is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Input Image")
                st.image(image, width=300)
            
            with col2:
                st.subheader("🎯 Predictions")
                
                # Make prediction
                predictions, final_prediction = predict_image(model, image, selected_model)
                
                if predictions is not None:
                    # Styled prediction display
                    st.markdown(f"### **🏆 Final Prediction: {final_prediction}**")
                    
                    st.write("**📊 Top 3 Predictions:**")
                    for i, pred in enumerate(predictions, 1):
                        confidence = pred['confidence']
                        if i == 1:  # Highest confidence
                            st.success(f"{i}. {pred['class']}: {confidence:.2f}%")
                        elif confidence > 10:
                            st.info(f"{i}. {pred['class']}: {confidence:.2f}%")
                        else:
                            st.write(f"{i}. {pred['class']}: {confidence:.2f}%")
                    
                    # for XAI
                    predicted_class_idx = CLASS_NAMES.index(final_prediction)

                else:
                    st.error(f"❌ Prediction failed: {final_prediction}")

            if predictions is not None:
                st.markdown("---")
                
                # layout for XAI header and button
                header_col1, header_col2 = st.columns([4, 1])
                with header_col1:
                    st.header("🎨 XAI Explanations")
                with header_col2:
                    generate_button = st.button("Generate Explanations", use_container_width=True)            
                if generate_button:
                    explanations_data = {}


                    with st.spinner("🔄 Generating CAM explanations..."):
                        explanations, error = generate_xai_explanations(
                            model, image, selected_model, predicted_class_idx
                        )
                    if explanations is not None:
                        # 5-column grid
                        cam_methods = ['GradCAM', 'GradCAM++', 'EigenCAM', 'AblationCAM']
                        for method_name in cam_methods:
                            if method_name in explanations:
                                explanations_data[method_name] = explanations[method_name]
                    else:
                        st.error(f"❌ CAM generation failed: {error}")


                    with st.spinner("🔄 Generating LIME explanation..."):
                        lime_result, lime_error = generate_lime_explanation(
                            model, image, selected_model, predicted_class_idx
                        )
                        if lime_result is not None:
                            explanations_data['LIME'] = lime_result
                        else:
                            st.error(f"❌ LIME generation failed: {lime_error}")

                            
                        if explanations_data:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            columns = [col1, col2, col3, col4, col5]
                            method_order = ['LIME', 'GradCAM', 'GradCAM++', 'EigenCAM', 'AblationCAM']

                            for i, method_name in enumerate(method_order):
                                if method_name in explanations_data:
                                    with columns[i]:
                                        st.markdown(f"**{method_name}**")
                                        st.image(explanations_data[method_name], use_container_width=True)

                    if explanations_data:
                        st.success(f"✅ Generated {len(explanations_data)} explanations successfully!")
                        st.info(f"💡 **Interpretation:** The highlighted regions show what the {selected_model} model focuses on to classify this image as '{final_prediction}'")


                        st.markdown("---")
                        download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
                        with download_col2:
                            # # Import the download function
                            # from model_utilities import create_download_zip


                            cam_explanations = {k: v for k, v in explanations_data.items() if k != 'LIME'}
                            lime_data = explanations_data.get('LIME', None)
                            zip_data = create_download_zip(
                                cam_explanations, 
                                lime_data, 
                                image, 
                                selected_model, 
                                final_prediction
                            )
                            if zip_data:
                                st.download_button(
                                    label="📥 Download Results (ZIP)",
                                    data=zip_data,
                                    file_name=f"plum_disease_results_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    type="secondary",
                                    use_container_width=True
                                )
                                st.caption("📦 Includes: Original image, CAM visualizations, LIME explanation, and summary report")
                            else:
                                st.error("❌ Failed to create download file")
    else:
        st.sidebar.error(f"❌ Failed to load model: {result}")


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🍃 Plum Disease Classification System | Built with Streamlit & PyTorch</p>
    </div>
    """, 
    unsafe_allow_html=True
)