import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml","setup.py"],
    pythonpath=True,
    dotenv=True,
)
import timm
import urllib
import torch
import wget
import numpy as np
import os.path
from omegaconf import OmegaConf
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image, ImageFont, ImageDraw, ImageOps

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)

from pytorch_lightning import LightningModule

from importlib import import_module

@utils.task_wrapper
def explain(cfg: DictConfig) -> Tuple[dict, dict]:
    log.info(f"Listing all techniques <{cfg.techniques}>")
    
    log.info(f"Initialising {cfg.model.model_name} model")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    model.eval()
    model = model.to(cfg.device)
    # checkpoint = torch.load("epoch_035.ckpt")
    # model.load_state_dict(checkpoint['state_dict'])

    url, filename = (
    cfg.labels.url,
    cfg.labels.file_path,
    )
    if not os.path.exists(cfg.labels.file_path):
        urllib.request.urlretrieve(url, filename)
    with open(cfg.labels.file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    if not os.path.exists(cfg.image_store_path):
        os.mkdir(cfg.image_store_path, mode = 0o777)
    
    images_metadata = []
    for i in cfg.image_urls:
        filename = i["sample"]["file_name"]
        url = i["sample"]["url"]
        if not os.path.exists(cfg.image_store_path + filename):
            filename = wget.download(url, out=cfg.image_store_path + filename)
        sample_details = OmegaConf.to_container(i, resolve=True)
        sample_details["sample"].update({"file_path" : cfg.image_store_path + filename})
        images_metadata.append(sample_details)
    
    cap_attr = import_module("captum.attr")
    from captum.attr import visualization as viz
    from captum.attr import IntegratedGradients
    from captum.attr import NoiseTunnel
    from captum.attr import Saliency
    from captum.attr import Occlusion


    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    from captum.robust import PGD

    import io


    def generate_gradients_image(img_tensor,target,model):
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(img_tensor, target=target, internal_batch_size=10,n_steps=200)


        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

        _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    method='heat_map',
                                    cmap=default_cmap,
                                    show_colorbar=True,
                                    sign='positive',
                                    outlier_perc=1,
                                    use_pyplot=False)
        buf = io.BytesIO()
        _[0].savefig(buf)
        buf.seek(0)
        return Image.open(buf)
    
    def generate_gradients_with_noise_image(img_tensor,target,model,transformed_img):
        torch.cuda.empty_cache()
        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_ig_nt = noise_tunnel.attribute(img_tensor, nt_samples=200, nt_type='smoothgrad_sq', 
                                                    nt_samples_batch_size=1,target=target)
        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["heat_map"],
                                            ["positive"],
                                            cmap=default_cmap,
                                            show_colorbar=True,
                                            use_pyplot=False)
        buf = io.BytesIO()
        _[0].savefig(buf)
        buf.seek(0)
        return Image.open(buf)

    def generate_occlusion_image(img_tensor,target,model,transformed_img):
        torch.cuda.empty_cache()
        occlusion = Occlusion(model)

        attributions_occ = occlusion.attribute(img_tensor,
                                            strides = (3, 8, 8),
                                            target=target,
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["heat_map"],
                                            ["positive"],
                                            show_colorbar=True,
                                            outlier_perc=2,
                                            )
        buf = io.BytesIO()
        _[0].savefig(buf)
        buf.seek(0)
        return Image.open(buf)
    
    def generate_gradcam_image(img_tensor,target,model,cuda_available):
        torch.cuda.empty_cache()        
        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                            height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        target_layers = [model.net.blocks[-1].norm1]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=cuda_available,
                reshape_transform=reshape_transform)

        targets = [ClassifierOutputTarget(target)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        img_tensor.requires_grad = True
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0, :]

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        inv_transform= T.Compose([
            T.Normalize(
                mean = (-1 * np.array(mean) / np.array(std)).tolist(),
                std = (1 / np.array(std)).tolist()
            ),
        ])

        rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return Image.fromarray(np.uint8(visualization)).convert('RGB')
    
    def generate_gradcam_plus_plus_image(img_tensor,target,model,cuda_available):
        torch.cuda.empty_cache()
        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                            height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        target_layers = [model.net.blocks[-1].norm1]

        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=cuda_available,
                reshape_transform=reshape_transform)

        targets = [ClassifierOutputTarget(target)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        img_tensor.requires_grad = True
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0, :]

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        inv_transform= T.Compose([
            T.Normalize(
                mean = (-1 * np.array(mean) / np.array(std)).tolist(),
                std = (1 / np.array(std)).tolist()
            ),
        ])

        rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return Image.fromarray(np.uint8(visualization)).convert('RGB')
    

    def get_prediction(model, categories,image: torch.Tensor,device):
        model = model.to(device)
        img_tensor = image.to(device)
        with torch.no_grad():
            output = model(img_tensor)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = categories[pred_label_idx.item()]

        return predicted_label, prediction_score.squeeze().item()


    def generate_pgd_image(model,img_tensor,device,categories):
        torch.cuda.empty_cache()
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        inv_transform= T.Compose([
            T.Normalize(
                mean = (-1 * np.array(mean) / np.array(std)).tolist(),
                std = (1 / np.array(std)).tolist()
            ),
        ])
        pgd = PGD(model, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1, upper_bound=1)  # construct the PGD attacker

        perturbed_image_pgd = pgd.perturb(inputs=img_tensor, radius=0.13, step_size=0.02, 
                                        step_num=7, target=torch.tensor([199]).to(device), targeted=True) 
        new_pred_pgd, score_pgd = get_prediction(model, categories,perturbed_image_pgd,device)
        regenerated_image = inv_transform(perturbed_image_pgd).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        return  Image.fromarray(np.uint8(regenerated_image * 255)).convert('RGB'),new_pred_pgd,score_pgd

    log.info(f"Starting to execute all techniques")
    techniques_to_use = cfg.techniques
    for sample in images_metadata:
        log.info(f"Sample <{sample['sample']['file_name']}>")
        img = Image.open(sample["sample"]["file_path"])

        transformed_img = model.train_transform(img)

        img_tensor = model.transform_normalize(transformed_img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(cfg.device)
        if "Original_Image" in techniques_to_use:
            
            sample["sample"]["Original_Image"] = Image.open(sample["sample"]["file_path"])
            log.info(f"Original image logged for sample <{sample['sample']['file_name']}>")
        if "IG" in techniques_to_use:
            sample["sample"]["IG"] = generate_gradients_image(img_tensor,
                    sample["sample"]["class_num"],model)
            log.info(f"Integrated Gradients image logged for sample <{sample['sample']['file_name']}>") 
        if "IG w/ Noise Tunnel" in techniques_to_use:
            sample["sample"]["IG w/ Noise Tunnel"] = generate_gradients_with_noise_image(img_tensor,
                    sample["sample"]["class_num"],model,transformed_img)
            log.info(f"IG w/ Noise Tunnel image logged for sample <{sample['sample']['file_name']}>")
        if "Occlusion" in techniques_to_use:
            sample["sample"]["Occlusion"] = generate_occlusion_image(img_tensor,
                    sample["sample"]["class_num"],model,transformed_img)
            log.info(f"Occlusion image logged for sample <{sample['sample']['file_name']}>")
        if "GradCAM" in techniques_to_use:
            cuda_available = True if cfg.device == "cuda" else False
            sample["sample"]["GradCAM"] = generate_gradcam_image(img_tensor,
                                            sample["sample"]["class_num"],model,cuda_available)
            log.info(f"GradCAM image logged for sample <{sample['sample']['file_name']}>")
        
        if "GradCAM++" in techniques_to_use:
            cuda_available = True if cfg.device == "cuda" else False
            sample["sample"]["GradCAM++"] = generate_gradcam_plus_plus_image(img_tensor,
                                            sample["sample"]["class_num"],model,cuda_available)
            log.info(f"GradCAM++ image logged for sample <{sample['sample']['file_name']}>")

        if cfg.PGD:
            sample["sample"]["PGD"],_,_ = generate_pgd_image(model,img_tensor,cfg.device,
                                            categories)
            log.info(f"PGD image logged for sample <{sample['sample']['file_name']}>")
        else:
            pass

    
    def image_grid(w, h, rows, cols):
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        return grid
    
    w,h = 224, 224
    rows = len(images_metadata)
    cols = len(techniques_to_use)
    grid = image_grid(w,h,rows,cols)

    for i,sample in enumerate(images_metadata):
        for j,techn in enumerate(techniques_to_use):
            grid.paste(sample["sample"][techn].resize((w,h)),box=(j%cols*w, i%cols*h))

    border = 30
    feature_explanation = ImageOps.expand(grid, border=border, fill=(255,255,255))
    draw = ImageDraw.Draw(feature_explanation)
    font = ImageFont.truetype("FONTS/arial.ttf", 15)
            

    for j,techn in enumerate(techniques_to_use):
        wi, hi = draw.textsize(techn,font)
        draw.text((w*j+((w-wi)/2)+border,0),techn,(0,0,0),font=font)




    
    w,h = 224, 224
    rows = len(images_metadata)
    cols = len(["Original_Image","PGD"])
    grid = image_grid(w,h,rows,cols)

    for i,sample in enumerate(images_metadata):
        for j,techn in enumerate(["Original_Image","PGD"]):
            grid.paste(sample["sample"][techn].resize((w,h)),box=(j%cols*w, i%cols*h))

    border = 30
    model_robustness = ImageOps.expand(grid, border=border, fill=(255,255,255))
    draw = ImageDraw.Draw(model_robustness)
    font = ImageFont.truetype("FONTS/arial.ttf", 15)
            

    for j,techn in enumerate(["Original_Image","PGD predict cat"]):
        wi, hi = draw.textsize(techn,font)
        draw.text((w*j+((w-wi)/2)+border,0),techn,(0,0,0),font=font)

    return {},{}


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    _ = explain(cfg)


if __name__ == "__main__":
    main()
