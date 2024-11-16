from ultralytics import YOLO

model = YOLO('best.pt')

img_size = 640
batch_size = 8
data_path = r"E:\Trial15BTP\data.yaml"

val_metrics = model.val(
    data=data_path,
    split='val',
    imgsz=img_size,
    batch=batch_size,
    device='cpu',
    verbose=True,
    plots=True
)


print("Validation Metrics:")
print(f"Precision (mean): {val_metrics.box.mean_results()[0]:.4f}")
print(f"Recall (mean): {val_metrics.box.mean_results()[1]:.4f}")
print(f"mAP@0.5 (mean): {val_metrics.box.mean_results()[2]:.4f}")
print(f"mAP@0.5:0.95 (mean): {val_metrics.box.mean_results()[3]:.4f}")


for idx, class_name in enumerate(model.names):
    class_metrics = val_metrics.box.class_result(idx)
    if class_metrics:  
        class_precision, class_recall, class_mAP50, class_mAP50_95 = class_metrics
        print(f"Class '{class_name}' (Class ID: {idx}):")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  Recall: {class_recall:.4f}")
        print(f"  mAP@0.5: {class_mAP50:.4f}")
        print(f"  mAP@0.5:0.95: {class_mAP50_95:.4f}")
        print("\n")
