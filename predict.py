import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings
import cv2
warnings.filterwarnings('ignore')

from unet import UNet

LR = 1e-5

def load_trained_model(checkpoint_path, device='auto'):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Загрузка модели на: {device}")
    
    if device.type == 'cpu':
        model = UNet.load_from_checkpoint(
            checkpoint_path,
            in_channels=3,
            out_channels=1,
            lr=LR,
            map_location='cpu'
        )
    else:
        model = UNet.load_from_checkpoint(
            checkpoint_path,
            in_channels=3,
            out_channels=1,
            lr=LR
        )
    
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path, image_size=256, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    image_tensor = transform(image)
    
    image_tensor = image_tensor.to(device)
    
    return image_tensor.unsqueeze(0), original_size

def predict_with_probabilities(model, image_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        binary_mask = (probabilities > threshold).float()
    
    probabilities = probabilities.squeeze().cpu().numpy()
    binary_mask = binary_mask.squeeze().cpu().numpy()
    
    return probabilities, binary_mask

def save_probability_map(prob_map, output_path, original_size=None, colormap='jet'):
    prob_map_255 = (prob_map * 255).astype(np.uint8)
    
    if colormap == 'jet':
        colored = cv2.applyColorMap(prob_map_255, cv2.COLORMAP_JET)
    elif colormap == 'hot':
        colored = cv2.applyColorMap(prob_map_255, cv2.COLORMAP_HOT)
    elif colormap == 'viridis':
        colored = cv2.applyColorMap(prob_map_255, cv2.COLORMAP_VIRIDIS)
    else:
        colored = cv2.applyColorMap(prob_map_255, cv2.COLORMAP_JET)
    
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    if original_size:
        colored_pil = Image.fromarray(colored_rgb)
        colored_pil = colored_pil.resize(original_size, Image.BILINEAR)
        colored_rgb = np.array(colored_pil)

    Image.fromarray(colored_rgb).save(output_path)
    return colored_rgb

def save_binary_mask(mask, output_path, original_size=None):
    if original_size:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
    else:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    
    mask_pil.save(output_path)
    return mask_pil

def visualize_all_results(image_path, probabilities, binary_mask, 
                         save_path=None, colormap='jet'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)
    
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Оригинал")
    axes[0, 0].axis('off')
    
    im_prob = axes[0, 1].imshow(probabilities, cmap=colormap)
    axes[0, 1].set_title("Вероятностная карта")
    axes[0, 1].axis('off')
    plt.colorbar(im_prob, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(binary_mask, cmap='gray')
    axes[0, 2].set_title(f"Бинарная маска (порог=0.5)")
    axes[0, 2].axis('off')
    
    from matplotlib.cm import get_cmap
    cmap = get_cmap(colormap)
    prob_colored = cmap(probabilities)
    
    axes[1, 0].imshow(original_array)
    axes[1, 0].imshow(prob_colored, alpha=0.6)
    axes[1, 0].set_title("наложение вероятностей")
    axes[1, 0].axis('off')
    
    overlay_binary = original_array.copy()

    overlay_binary[binary_mask > 0.5] = [255, 0, 0]
    axes[1, 1].imshow(overlay_binary)
    axes[1, 1].set_title("наложение маски")
    axes[1, 1].axis('off')
    
    axes[1, 2].hist(probabilities.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 2].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
    axes[1, 2].set_title("Рспределение вероятности")
    axes[1, 2].set_xlabel("вероятность")
    axes[1, 2].set_ylabel("частота")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"визуализация сохранена: {save_path}")
    
    plt.show()

def resize_to_original(data, original_size, interpolation=Image.BILINEAR):
    if isinstance(data, np.ndarray):
        data_pil = Image.fromarray((data * 255).astype(np.uint8))
        data_pil = data_pil.resize(original_size, interpolation)
        return np.array(data_pil) / 255.0
    return data

def predict_single_image_with_probs(image_path, checkpoint_path, 
                                   output_prob_path="probability_map.png",
                                   output_mask_path="binary_mask.png",
                                   output_visualization_path="full_visualization.png",
                                   threshold=0.5,
                                   device='auto',
                                   colormap='jet'):

    print(f"Обработка: {image_path}")
    
    model, device_obj = load_trained_model(checkpoint_path, device)
    
    image_tensor, original_size = preprocess_image(image_path, device=device_obj)
    
    probabilities, binary_mask = predict_with_probabilities(model, image_tensor, threshold)

    prob_resized = resize_to_original(probabilities, original_size, Image.BILINEAR)
    mask_resized = resize_to_original(binary_mask, original_size, Image.NEAREST)
    
    if output_prob_path:
        prob_colored = save_probability_map(
            prob_resized, 
            output_prob_path, 
            original_size, 
            colormap
        )
        print(f"карта сохранена: {output_prob_path}")
    
    if output_mask_path:
        save_binary_mask(mask_resized, output_mask_path)
        print(f"маска сохранена: {output_mask_path}")
    
    if output_visualization_path:
        visualize_all_results(
            image_path, 
            prob_resized, 
            mask_resized, 
            output_visualization_path,
            colormap
        )
    else:
        visualize_all_results(image_path, prob_resized, mask_resized, colormap=colormap)

    print("\n" + "="*50)
    print("Статистика предсказания:")
    print("="*50)
    print(f"Разброс вероятности: [{prob_resized.min():.4f}, {prob_resized.max():.4f}]")
    print(f"Покрытие маской: {mask_resized.mean():.2%}")
    print(f"Пикселей выше порога: {(prob_resized > threshold).sum()} / {prob_resized.size}")
    print("="*50)
    
    return prob_resized, mask_resized

def get_probabilities_only(image_path, checkpoint_path, device='auto'):
    model, device_obj = load_trained_model(checkpoint_path, device)
    image_tensor, original_size = preprocess_image(image_path, device=device_obj)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    prob_resized = resize_to_original(probabilities, original_size, Image.BILINEAR)
    return prob_resized

if __name__ == "__main__":

    CHECKPOINT_PATH = "./checkpoints/last.ckpt"
    IMAGE_PATH = "test.png"

    probabilities, binary_mask = predict_single_image_with_probs(
        image_path=IMAGE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_prob_path="probability_heatmap.png",
        output_mask_path="binary_mask.png",
        output_visualization_path="full_prediction_results.png",
        threshold=0.5,
        device='auto',
        colormap='jet'
    )
    
