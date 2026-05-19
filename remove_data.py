from pathlib import Path

ORIGEN = Path("DOTA_COMPLET")

N = 8

extensiones = {".png", ".jpg", ".jpeg", ".webp"}

for tipo in ["fake", "real"]:

    raiz = ORIGEN / tipo

    if not raiz.exists():
        continue

    for carpeta in raiz.rglob("*"):

        if not carpeta.is_dir():
            continue

        relativa = carpeta.relative_to(raiz)
        profundidad = len(relativa.parts)

       

        if tipo == "fake" and profundidad != 2:
            continue

        if tipo == "real" and profundidad != 1:
            continue

        imagenes = sorted([
            img for img in carpeta.iterdir()
            if img.is_file() and img.suffix.lower() in extensiones
        ])

        if len(imagenes) == 0:
            continue

      
        borrar = imagenes[-N:]

        for img in borrar:
            img.unlink()

        print(f"{carpeta}: eliminadas {len(borrar)} imágenes")