import io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load face detection and embedding models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=240, post_process=True, keep_all=True, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load average embeddings for known identities
embedding_data = torch.load("embeddings/embeddings.pt", map_location=torch.device("cpu"))

def locate_faces(image):
    """Detect faces and return bounding boxes, probabilities, and cropped faces."""
    cropped_faces, probs = mtcnn(image, return_prob=True)
    boxes, _ = mtcnn.detect(image)
    if boxes is None or cropped_faces is None:
        return []
    return list(zip(boxes, probs, cropped_faces))

def determine_name_dist(cropped_image, threshold=0.9):
    """Identify the closest known identity for a cropped face embedding."""
    emb = resnet(cropped_image.unsqueeze(0))
    distances = [(torch.dist(emb, known_emb).item(), name) for name, known_emb in embedding_data.items()]
    dist, closest = min(distances)
    name = closest if dist < threshold else "Unrecognized"
    return name, dist

def add_labels_to_image(image):
    """
    Detect faces, draw bounding boxes, and label each detected face.
    Label is centered below the bounding box.
    """
    width, height = image.width, image.height
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

    faces = locate_faces(image)
    if not faces:
        return None

    for box, prob, cropped in faces:
        if prob < 0.9:
            continue
        name, dist = determine_name_dist(cropped)
        box_color = "red" if name == "Unrecognized" else "blue"

        # Draw bounding box
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=box_color,
            facecolor='none'
        )
        axis.add_patch(rect)

        # Draw label below the bounding box, centered
        label = f"{name}"
        center_x = (x1 + x2) / 2
        label_y = y2 + 15
        axis.text(
            center_x,
            label_y,
            label,
            fontsize=10,
            color="white",
            weight='bold',
            ha='center',
            va='center',
            bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round,pad=0.3'),
            clip_on=True
        )

    return fig

def matplotlib_to_bytes(fig, output_path):
    """Save matplotlib figure to file and return as bytes buffer."""
    buf = io.BytesIO()
    plt.savefig(output_path)
    plt.close()
    buf.seek(0)
    return buf