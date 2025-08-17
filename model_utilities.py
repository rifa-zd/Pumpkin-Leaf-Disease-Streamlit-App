import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image

from skimage.segmentation import mark_boundaries
from datetime import datetime



class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 256//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256//16, 256, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)


        # Class names


CLASS_NAMES = ['Dead Leaf', 'Healthy Fruit', 'Healthy Leaf', 'Insect Hole', 'Unhealthy Fruit', 'Yellow']

# Model configurations (metadata)

MODEL_CONFIGS = {
    'CustomCNN': {
        'architecture': 'Custom CNN',
        'input_size': (224, 224),
        'classes': len(CLASS_NAMES),
        'checkpoint_path': 'models/custom_cnn_non_weighted.pt'
    },
    'CustomCNN_Weighted': {
        'architecture': 'Custom CNN with Class Weights in CE Loss',
        'input_size': (224, 224),
        'classes': len(CLASS_NAMES),
        'checkpoint_path': 'models/custom_cnn_weighted.pt'
    },
    'ResNet50': {
        'architecture': 'ResNet50 (Transfer Learning)',
        'input_size': (224, 224), 
        'classes': len(CLASS_NAMES),
        'checkpoint_path': 'models/TL_resnet50.pt'
    },
    'InceptionV3': {
        'architecture': 'InceptionV3 (Transfer Learning)',
        'input_size': (299, 299),
        'classes': len(CLASS_NAMES), 
        'checkpoint_path': 'models/TL_inception_v3.pt'
    },
    'MobileNetV2': {
        'architecture': 'MobileNetV2 (Transfer Learning)',
        'input_size': (224, 224),
        'classes': len(CLASS_NAMES),
        'checkpoint_path': 'models/TL_mobilenet_v2.pt'
    },
    'VGG16': {
        'architecture': 'VGG16 (Transfer Learning)', 
        'input_size': (224, 224),
        'classes': len(CLASS_NAMES),
        'checkpoint_path': 'models/TL_vgg16.pt'
    }
}


def load_model(model_name):
    """Loading a specific model by name"""
    try:
        config = MODEL_CONFIGS[model_name]
        num_classes = config['classes']
        checkpoint_path = config['checkpoint_path']
        
        if model_name == 'CustomCNN':
            model = CustomCNN(num_classes)
        elif model_name == 'CustomCNN_Weighted':
            model = CustomCNN(num_classes)
        elif model_name == 'ResNet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'InceptionV3':
            model = models.inception_v3(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'MobileNetV2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'VGG16':
            model = models.vgg16(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
        # Load weights
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        model.eval()
        
        return model, config
    except Exception as e:
        return None, str(e)
    

    # Image preprocessing transforms


def get_transforms(model_name):
    """Get the appropriate transforms for each model"""
    if model_name in ['CustomCNN', 'CustomCNN_Weighted']:
        # Custom normalization for CustomCNN
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.685, 0.693, 0.665],
                std=[0.175, 0.166, 0.223]
            )
        ])
    elif model_name == 'InceptionV3':
        # Custom size for InceptionV3
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # ImageNet normalization for transfer learning models except the above
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    

def predict_image(model, image, model_name):
    """Make prediction on an image"""
    try:
        # Get appropriate transforms
        transform = get_transforms(model_name)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0)  
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            class_name = CLASS_NAMES[top3_indices[i]]
            confidence = top3_prob[i].item() * 100
            predictions.append({
                'class': class_name,
                'confidence': confidence
            })
        
        return predictions, CLASS_NAMES[top3_indices[0]]  
        
    except Exception as e:
        return None, str(e) 
    

def get_unnormalize_transform(model_name):
    """Get unnormalization transform for each model"""
    if model_name in ['CustomCNN', 'CustomCNN_Weighted']:
        return transforms.Normalize(
            mean=[-0.685/0.175, -0.693/0.166, -0.665/0.223],
            std=[1/0.175, 1/0.166, 1/0.223]
        )
    else:  # Transfer learning models
        return transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )


def get_target_layers(model, model_name):
    """Get target layers for different models"""
    if model_name in ['CustomCNN', 'CustomCNN_Weighted']:
    # if model_name == 'CustomCNN':
        return [model.conv4[0]]  # Last conv layer in my cse conv4
    # elif model_name == 'CustomCNN_Weighted':
    #     return [model.conv4[0]]
    elif model_name == 'ResNet50':
        return [model.layer4[-1]]
    elif model_name == 'InceptionV3':
        return [model.Mixed_7c]
    elif model_name == 'MobileNetV2':
        return [model.features[-1]]
    elif model_name == 'VGG16':
        return [model.features[-1]]
    else:
        return None
    


def generate_xai_explanations(model, image, model_name, predicted_class_idx):
    """Generate all XAI explanations with proper error handling"""
    try:
        # Get transforms and prepare input
        transform = get_transforms(model_name)
        unnormalize = get_unnormalize_transform(model_name)
        
        # Convert PIL to tensor and add batch dimension
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to reduce processing time
        image_resized = image.resize((224, 224))
        input_tensor = transform(image_resized).unsqueeze(0)
        
        input_tensor_cpu = input_tensor.squeeze(0)
        unnormalized_image = unnormalize(input_tensor_cpu)
        original_image_np = unnormalized_image.permute(1, 2, 0).numpy()
        original_image_np = np.clip(original_image_np, 0, 1)
        
        # Get target layers
        target_layers = get_target_layers(model, model_name)
        if target_layers is None:
            return None, "Target layers not found for this model"
        
        target = [ClassifierOutputTarget(predicted_class_idx)]
        results = {'original': (original_image_np * 255).astype(np.uint8)}
        
        cam_methods = [
            ('GradCAM', GradCAM),
            ('GradCAM++', GradCAMPlusPlus),
            ('EigenCAM', EigenCAM),
            ('AblationCAM', AblationCAM)
        ]
        
        for method_name, cam_class in cam_methods:
            try:
                print(f"Generating {method_name}...")
                cam = cam_class(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)
                
                if grayscale_cam is not None and len(grayscale_cam.shape) > 0:
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
                    results[method_name] = cam_image
                    print(f"{method_name} generated successfully")
                else:
                    print(f"{method_name} returned empty result")
                    
                del cam
                    
            except Exception as e:
                print(f"{method_name} failed: {e}")
                continue
        
        print(f"Generated methods: {list(results.keys())}")
        return results, None
        
    except Exception as e:
        print(f"Overall error: {e}")
        return None, str(e)

def generate_lime_explanation(model, image, model_name, predicted_class_idx):
    """Generate LIME explanation with performance optimizations"""
    try:
        transform = get_transforms(model_name)
        unnormalize = get_unnormalize_transform(model_name)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_resized = image.resize((224, 224))
        input_tensor = transform(image_resized)
        unnormalized_image = unnormalize(input_tensor)
        original_image_np = unnormalized_image.permute(1, 2, 0).numpy()
        original_image_np = np.clip(original_image_np, 0, 1)
        
        def batch_predict(images):
            model.eval()
            batch_size = min(10, len(images))
            results = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_tensors = torch.stack([
                    transform(Image.fromarray((img * 255).astype(np.uint8))) 
                    for img in batch_images
                ])
                
                with torch.no_grad():
                    logits = model(batch_tensors)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    results.append(probs.numpy())
            
            return np.vstack(results)
        
        explainer = lime_image.LimeImageExplainer()
        lime_explanation = explainer.explain_instance(
            original_image_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=30,  
            num_features=100000,  
            batch_size=10
        )
        
        lime_image_result, lime_mask = lime_explanation.get_image_and_mask(
            label=predicted_class_idx,
            positive_only=True,
            hide_rest=False,
            num_features=8, 
            min_weight=0.01
        )
        
        lime_image_result = mark_boundaries(lime_image_result, lime_mask)
        
        return lime_image_result, None
        
    except Exception as e:
        return None, str(e)



def create_download_zip(explanations, lime_result, original_image, model_name, prediction):
    """Create a ZIP file with all visualizations for download"""
    try:
        import zipfile
        import io
        from PIL import Image as PILImage
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save original image
            original_buffer = io.BytesIO()
            original_image.save(original_buffer, format='PNG')
            zip_file.writestr(f"original_image.png", original_buffer.getvalue())
            
            # Save CAM results
            if explanations:
                for method_name, result_image in explanations.items():
                    if method_name != 'original' and result_image is not None:
                        img_buffer = io.BytesIO()
                        if isinstance(result_image, np.ndarray):
                            pil_img = PILImage.fromarray(result_image.astype('uint8'))
                            pil_img.save(img_buffer, format='PNG')
                            zip_file.writestr(f"{method_name}_{model_name}.png", img_buffer.getvalue())
            
            # Save LIME result
            if lime_result is not None:
                lime_buffer = io.BytesIO()
                lime_pil = PILImage.fromarray((lime_result * 255).astype('uint8'))
                lime_pil.save(lime_buffer, format='PNG')
                zip_file.writestr(f"LIME_{model_name}.png", lime_buffer.getvalue())
            
            # Create a summary text file
            summary = f"""Plum Disease Classification Results

Model: {model_name}
Prediction: {prediction}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files included:
- original_image.png: Input image"""

            if explanations:
                for method in explanations.keys():
                    if method != 'original':
                        summary += f"\n- {method}_{model_name}.png: {method} visualization"
            
            if lime_result is not None:
                summary += f"\n- LIME_{model_name}.png: LIME explanation"
                        
            zip_file.writestr("summary.txt", summary)
        
        return zip_buffer.getvalue()
        
    except Exception as e:
        print(f"Download creation error: {e}")
        return None
    



