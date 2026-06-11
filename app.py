import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
from collections import Counter

st.set_page_config(page_title="Bacteria FSL", layout="centered")
st.title("Few-Shot Classification (ProtoNet)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "advanced_protonet_resnet18.pth"
SUPPORT_DIR = "FSL_Dataset/all_classes"
NUM_SUPPORT_PER_CLASS = 5

# ---- 1. හරිම Model Architecture එක (Colab එකේ තිබ්බ එකමයි) ----
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        # හිස් මොඩල් එකක් අරන් අපේ පරණ Weights ටික මේකට දානවා
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# ---- Utilities ----
def get_image_paths_for_class(class_dir, n=5):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(class_dir, e)))
    paths = sorted(paths)[:n]
    return paths

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@st.cache_resource
def load_model(model_path):
    m = ProtoNet().to(DEVICE)
    try:
        # දැන් හරියටම මැච් වෙන නිසා strict=False ඕන නෑ, Weights 100% ලෝඩ් වෙනවා!
        state = torch.load(model_path, map_location=DEVICE)
        m.load_state_dict(state)
    except Exception as e:
        st.error(f"Weights ලෝඩ් වුණේ නෑ!: {e}")
    m.eval()
    return m

def image_to_embedding(image: Image.Image, model):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()

@st.cache_resource
def build_prototypes(support_dir, _model, k_per_class=5):
    prototypes = {}
    classes = sorted([d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))])
    for cls in classes:
        class_dir = os.path.join(support_dir, cls)
        paths = get_image_paths_for_class(class_dir, n=k_per_class)
        if len(paths) == 0:
            continue
        embs = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
                emb = image_to_embedding(im, _model)
                embs.append(emb.numpy())
            except Exception as e:
                pass
        if len(embs) > 0:
            embs = np.stack(embs, axis=0)
            proto = embs.mean(axis=0)
            prototypes[cls] = proto
    return prototypes

# ---- Load model and prototypes ----
model = load_model(MODEL_PATH)
prototypes = build_prototypes(SUPPORT_DIR, model, NUM_SUPPORT_PER_CLASS)

if len(prototypes) == 0:
    st.error("No class prototypes found. Check SUPPORT_DIR contents.")
    st.stop()

st.success(f"✅ Loaded {len(prototypes)} class prototypes successfully with Trained Weights!")

# ---- Uploader ----


# ---- Uploader (Multiple Files) ----
# accept_multiple_files=True දාලා තියෙන නිසා දැන් පින්තූර ගොඩක් එකපාර සිලෙක්ට් කරන්න පුළුවන්
uploaded_files = st.file_uploader("Upload test images (Select multiple)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"⏳ Processing {len(uploaded_files)} images... Please wait.")
    all_preds = []
    
    # 1. හැම පින්තූරයක්ම එකින් එක රන් කරලා උත්තර ටික එකතු කරගන්නවා
    for uploaded_file in uploaded_files:
        try:
            test_img = Image.open(uploaded_file).convert("RGB")
            test_emb = image_to_embedding(test_img, model).numpy()

            class_names = list(prototypes.keys())
            proto_matrix = np.stack([prototypes[c] for c in class_names], axis=0)
            dists = np.linalg.norm(proto_matrix - test_emb[None, :], axis=1)
            min_idx = int(np.argmin(dists))
            pred_class = class_names[min_idx]
            
            all_preds.append(pred_class)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # 2. Majority Voting (වැඩිම වාර ගණනක් ආපු උත්තරේ තේරීම)
    if all_preds:
        vote_counts = Counter(all_preds)
        best_answer, max_votes = vote_counts.most_common(1)[0]
        
        # 3. අවසාන ප්‍රතිඵලය ලස්සනට පෙන්වීම
        st.success(f"🦠 Final Predicted Bacteria: **{best_answer}**")
        st.write(f"*(Confidence: {max_votes} out of {len(all_preds)} images agreed on this result)*")
        
        # 4. එකින් එක පින්තූරෙට ආපු උත්තර බලාගන්න වෙනම Expander එකක්
        with st.expander("📊 View Detailed Voting Results"):
            st.write("Results According to the images:")
            for name, count in vote_counts.items():
                # ප්‍රතිශතයත් එක්ක පෙන්වමු
                percentage = (count / len(all_preds)) * 100
                st.write(f"- **{name}**: {count} images ({percentage:.1f}%)")
                
            # අප්ලෝඩ් කරපු පින්තූර ටිකත් පොඩියට පෙන්වමු
            st.write("---")
            st.write("Uploaded Images:")
            cols = st.columns(len(uploaded_files[:5])) # මුල් පින්තූර 5 විතරක් පෙන්වමු UI එක ලස්සන වෙන්න
            for i, col in enumerate(cols):
                img = Image.open(uploaded_files[i])
                col.image(img, use_column_width=True)
            if len(uploaded_files) > 5:
                st.write(f"... and {len(uploaded_files) - 5} more images.")