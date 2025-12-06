# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 07:09:52 2025

@author: frederic.ros
"""
import os
from PIL import Image

def create_image_grid(directory, rows, cols, output_path="output_grid.jpg"):
    # Lister les fichiers image dans le répertoire
    print("directory", directory)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith(image_extensions)]
    
    # Vérification
    if len(image_paths) < rows * cols:
        raise ValueError("Not enough images in the directory to fill the grid.")

    # Charger les premières images nécessaires
    selected_images = [Image.open(img_path) for img_path in image_paths[:rows * cols]]

    # Option 1 : redimensionner toutes les images à la taille de la plus petite
    min_width = min(img.size[0] for img in selected_images)
    min_height = min(img.size[1] for img in selected_images)
    resized_images = [img.resize((min_width, min_height)) for img in selected_images]

    # Créer l'image finale
    grid_img = Image.new('RGB', (cols * min_width, rows * min_height))

    for idx, img in enumerate(resized_images):
        row, col = divmod(idx, cols)
        x = col * min_width
        y = row * min_height
        grid_img.paste(img, (x, y))

    # Sauvegarde
    grid_img.save(output_path)
    print(f"✅ Grid image saved to {output_path}")

#directory = "resultetGraphe/ILLUSTRATIONSYNTHETIQUE/DATA2"
#directory = "resultetGraphe/ILLUSTRATIONSYNTHETIQUE/DATA3"
#directory = "resultetGraphe/REALALLAPRIL/SequencesAdvanced/SummarizeFigure"
#directory = "resultetGraphe/HIGHREAL/synthesegraphe"
#directory = "resultetGraphe/REALFINAL/synthesegraphique"
#directory = "resultetGraphe/SYNTHETIQUEGRAPHES0.8"
#directory = "resultetGraphe/GRAPHESYNTHETIQUE/novel"
#directory = "resultetGraphe/GRAPHEOVERVIEW/unique"
#directory = "resultsynthetique/our/ablation/MembershipetNoise"
#directory = "resultsynthetique/our/Testnoise/graphe"
#directory = "resultsynthetique/our/rank"
#directory = "../resultsynthetique/our/Highdimension/fusion2"
#directory = "../resultsreal/mediumdim/graphes/Fusion"
#directory = "../resultsreal/mediumdim/graphes/Fusion2"
#directory = "../resultsreal/mediumdim/graphes/Fusion3/Fusion"
#directory = "../resultsreal/mediumdim/graphes/FusionG"
#directory = "../resultsynthetique/our/training"
directory = "../resultsynthetique/our/Projection10to2"

rows = 1
cols = 3
#create_image_grid(directory, rows, cols, output_path="output_gridHighReal.jpg")
create_image_grid(directory, rows, cols, output_path="TrainingProjection.jpg")

