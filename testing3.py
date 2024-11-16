import cv2
from ultralytics import YOLO


model = YOLO(r'E:\Trial15BTP\runs\detect\train32222\weights\best.pt')


image1_path = r'E:\Trial15BTP\test_main\item1.jpg'  #single item image
image2_path = r'E:\Trial15BTP\test_main\items5.jpg'  #a group of items in iamges


results_image1 = model(image1_path)


if results_image1[0].boxes:
    class_id = int(results_image1[0].boxes[0].cls[0])
    class_name = model.names[class_id]
    print(f"Item in image1 is of class '{class_name}' (class ID: {class_id})")
else:
    print("No objects detected in image1.")
    exit()

results_image2 = model(image2_path)

found = False
for result in results_image2:
    boxes = result.boxes
    for box in boxes:
        detected_class_id = int(box.cls[0])
        if detected_class_id == class_id:
            found = True
            annotated_image = result.plot()
            cv2.imwrite('image2_with_detections.jpg', annotated_image)
            print(f"Item found in image2. Bounding box saved to 'image2_with_detections.jpg'.")
            break
    if found:
        break

if not found:
    print("Item from image1 not found in image2.")
