# predict_wallet.py
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs/detect/train2/weights/best.pt")
    src = r"dataset_wallet/images/val/teste.mp4"

    # show=True abre a janela com a detecção;
    # save=True também salva a imagem anotada em runs/detect/predict/
    results = model.predict(source=src, imgsz=640, conf=0.25, device="cpu", show=True, save=True)
