#!/usr/bin/env python3
"""Clasifica una ecografía mamaria con un ViT afinado en BUSI/Mendeley, en local.

Experimento: NO es una herramienta diagnóstica.

Uso:
    python clasificar.py RUTA_IMAGEN [--modelo ID_O_CARPETA]
"""
import argparse
import sys

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODELO_POR_DEFECTO = "sidushdid/ViT-base-patch16-BUSI-Mendeley-Adafactor"


def clasificar(ruta_imagen, modelo=MODELO_POR_DEFECTO):
    """Devuelve [(etiqueta, probabilidad), ...] ordenado de mayor a menor."""
    processor = AutoImageProcessor.from_pretrained(modelo)
    model = AutoModelForImageClassification.from_pretrained(modelo)
    model.eval()

    imagen = Image.open(ruta_imagen).convert("RGB")
    entradas = processor(images=imagen, return_tensors="pt")

    with torch.no_grad():
        logits = model(**entradas).logits
    probs = torch.softmax(logits, dim=-1)[0]

    resultados = [(model.config.id2label[i], p.item()) for i, p in enumerate(probs)]
    return sorted(resultados, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("imagen", help="ruta a la imagen (jpg, png, ...)")
    parser.add_argument("--modelo", default=MODELO_POR_DEFECTO,
                        help="id de Hugging Face o carpeta local del modelo")
    args = parser.parse_args()

    try:
        resultados = clasificar(args.imagen, args.modelo)
    except FileNotFoundError:
        sys.exit(f"No se encontró la imagen: {args.imagen}")

    print(f"Imagen: {args.imagen}")
    print(f"Modelo: {args.modelo}\n")
    for etiqueta, prob in resultados:
        print(f"  {etiqueta:<10} {prob:7.2%}")
    print("\nAviso: experimento con un modelo entrenado en datasets pequeños "
          "(BUSI/Mendeley). No es un diagnóstico.")


if __name__ == "__main__":
    main()
