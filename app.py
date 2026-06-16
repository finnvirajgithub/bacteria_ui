import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
import cv2
from collections import Counter

st.set_page_config(page_title="Automated Pathogen System", layout="wide")

# --- UI Header ---
st.title("🦠 Automated Light Microscopy Pathogen System")
st.markdown("Integrated framework for **Pathogen Identification (ProtoNet)** and **Edge-Based Colony Counting**.")

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "advanced_protonet_resnet18.pth"
SUPPORT_DIR = "Inbuilt_Support"
NUM_SUPPORT_PER_CLASS = 5

# ==========================================
# BACTERIA KNOWLEDGE BASE
# ==========================================
BACTERIA_INFO = {
    "Acinetobacter.baumanii": {"description": "Gram-negative, strictly aerobic, pleomorphic coccobacillus.", "diseases": "Hospital-acquired pneumonia, Bloodstream infections.", "details": "Highly associated with hospital-acquired (nosocomial) infections."},
    "Bacteroides.fragilis": {"description": "Gram-negative, obligately anaerobic, rod-shaped bacterium.", "diseases": "Intra-abdominal infections, Pelvic inflammatory disease.", "details": "Part of the normal human colon flora but becomes highly pathogenic if displaced."},
    "Enterococcus.faecalis": {"description": "Gram-positive, commensal bacterium inhabiting the gastrointestinal tracts.", "diseases": "Urinary Tract Infections (UTIs), Endocarditis.", "details": "High natural resistance to many antibiotics."},
    "Escherichia.coli": {"description": "Gram-negative, facultatively anaerobic, rod-shaped coliform bacterium.", "diseases": "Urinary Tract Infections (UTIs), Gastroenteritis.", "details": "Pathogenic strains can cause severe foodborne illnesses."},
    "Listeria.monocytogenes": {"description": "Gram-positive, rod-shaped bacterium.", "diseases": "Listeriosis, Meningitis.", "details": "A dangerous foodborne pathogen that can grow at refrigerator temperatures."},
    "Neisseria.gonorrhoeae": {"description": "Gram-negative diplococci bacteria.", "diseases": "Gonorrhea, Pelvic Inflammatory Disease (PID).", "details": "A sexually transmitted pathogen showing increasing antibiotic resistance."},
    "Proteus": {"description": "Gram-negative, facultatively anaerobic, rod-shaped bacterium.", "diseases": "Complicated UTIs, Kidney stones.", "details": "Produces urease, leading to rapid formation of kidney stones."},
    "Pseudomonas.aeruginosa": {"description": "Gram-negative, aerobic, rod-shaped bacterium.", "diseases": "Ventilator-associated pneumonia, Sepsis.", "details": "Known for distinctive green-blue pigment and high antibiotic resistance."},
    "Staphylococcus.aureus": {"description": "Gram-positive, round-shaped bacterium.", "diseases": "Skin infections, Pneumonia, Osteomyelitis.", "details": "Forms golden-yellow colonies. MRSA is a major healthcare concern."},
    "Staphylococcus.epidermidis": {"description": "Gram-positive bacterium, common skin flora.", "diseases": "Infections of intravascular devices.", "details": "Forms thick biofilms on plastic implants and catheters."},
    "Streptococcus.agalactiae": {"description": "Gram-positive coccus that forms chains (GBS).", "diseases": "Neonatal meningitis, Pneumonia.", "details": "Highly dangerous to infants during childbirth."},
    "Streptococcus.pneumoniae": {"description": "Gram-positive, lancet-shaped diplococci.", "diseases": "Community-acquired pneumonia, Meningitis.", "details": "Possesses a thick capsule making it resistant to immune cells."},
    "Candida.albicans": {"description": "An opportunistic pathogenic yeast (fungus).", "diseases": "Oral thrush, Invasive candidiasis.", "details": "Can overgrow and cause severe infections in immunocompromised individuals."}
}

# ---- 1. Model Architecture ----
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
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
    m = ProtoNet().to(torch.device("cpu"))
    try:
        state = torch.load(model_path, map_location=torch.device("cpu"))
        m.load_state_dict(state)
    except Exception as e:
        st.error(f"Weights load error (Ignore if just testing UI): {e}")
    m.eval()
    return m

def image_to_embedding(image: Image.Image, model):
    img = transform(image).unsqueeze(0).to(torch.device("cpu"))
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0)

@st.cache_resource
def build_prototypes(support_dir, _model, k_per_class=5):
    prototypes = {}
    os.makedirs(support_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))])
    
    if not classes:
        return prototypes
        
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
            except Exception:
                pass
        if len(embs) > 0:
            embs = np.stack(embs, axis=0)
            proto = embs.mean(axis=0)
            prototypes[cls] = proto
            
    return prototypes

# ---- Load model and prototypes ----
model = load_model(MODEL_PATH)
prototypes = build_prototypes(SUPPORT_DIR, model, NUM_SUPPORT_PER_CLASS)


# ==========================================
# Sidebar: Settings & Support Set Updater
# ==========================================
st.sidebar.header("🎛️ Advanced Settings")
THRESHOLD = st.sidebar.slider(
    "OOD Detection Threshold", 
    min_value=0.0, 
    max_value=80.0, 
    value=5.0, 
    step=1.0, 
    help="Lower values are stricter. If a sample's distance is higher than this, it is flagged as Unknown."
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Support Set Manager")
new_class_name = st.sidebar.text_input("New Species Name:")
new_class_files = st.sidebar.file_uploader("Upload exactly 5 images (5-Shot)", accept_multiple_files=True, type=["jpg","jpeg","png"])

if st.sidebar.button("Update Prototype Memory"):
    if len(new_class_files) != 5:
        st.sidebar.error("⚠️ Please upload exactly 5 images.")
    elif not new_class_name:
        st.sidebar.error("⚠️ Please enter a class name.")
    else:
        new_dir = os.path.join(SUPPORT_DIR, new_class_name)
        os.makedirs(new_dir, exist_ok=True)
        for f in new_class_files:
            with open(os.path.join(new_dir, f.name), "wb") as out_file:
                out_file.write(f.getbuffer())
        st.cache_resource.clear()
        st.sidebar.success(f"✅ '{new_class_name}' added! Refreshing...")
        st.rerun()

# ==========================================
# TABS CREATION
# ==========================================
tab1, tab2 = st.tabs(["🔬 Pathogen Identification (Few-Shot)", "🧮 Edge-Based Colony Counting"])

# ---------------------------------------------------------
# TAB 1: PATHOGEN IDENTIFICATION
# ---------------------------------------------------------
with tab1:
    if len(prototypes) == 0:
        st.warning("No class prototypes found in memory. Please use the sidebar to add a new support set.")
    else:
        st.success(f"✅ System Online: {len(prototypes)} distinct bacterial species loaded into Prototype Memory.")

        uploaded_files = st.file_uploader("📤 Upload microscopy images for analysis (Minimum 5 required)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True, key="query")

        if uploaded_files:
            if len(uploaded_files) < 5:
                st.warning(f"⚠️ You have uploaded {len(uploaded_files)} image(s). Please upload at least 5 images for Majority Voting.")
            else:
                st.info("⏳ Processing images via ResNet18... Please wait.")
                
                all_preds = []
                image_details = [] # List to save the distance values for each image
                
                for uploaded_file in uploaded_files:
                    try:
                        test_img = Image.open(uploaded_file).convert("RGB")
                        test_emb = image_to_embedding(test_img, model).numpy()

                        class_names = list(prototypes.keys())
                        proto_matrix = np.stack([prototypes[c] for c in class_names], axis=0)
                        
                        # Calculate Euclidean distance
                        dists = np.linalg.norm(proto_matrix - test_emb[None, :], axis=1)
                        min_idx = int(np.argmin(dists))
                        min_dist = float(dists[min_idx]) # Get the exact minimum distance value
                        
                        if min_dist > THRESHOLD:
                            pred_label = "Unknown Species (Out of Distribution)"
                        else:
                            pred_label = class_names[min_idx]
                            
                        all_preds.append(pred_label)
                        
                        # Save all prediction details as a dictionary
                        image_details.append({
                            "file": uploaded_file,
                            "pred": pred_label,
                            "dist": min_dist
                        })
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                if all_preds:
                    vote_counts = Counter(all_preds)
                    best_answer, max_votes = vote_counts.most_common(1)[0]
                    
                    st.markdown("---")
                    
                    if "Unknown Species" in best_answer:
                        st.error(f"🚨 **ALERT:** High-distance anomaly detected.\n\n**Final Prediction:** {best_answer}")
                    else:
                        st.success(f"🦠 **Final Predicted Bacteria:** {best_answer}")
                        st.write(f"*(Confidence: {max_votes} out of {len(all_preds)} images agreed on this result)*")
                        
                        st.markdown("### 📋 Clinical Information")
                        info = BACTERIA_INFO.get(best_answer)
                        if info:
                            st.info(f"**🔬 Description:** {info['description']}")
                            st.warning(f"**⚠️ Associated Diseases:** {info['diseases']}")
                            st.success(f"**💡 Clinical Details:** {info['details']}")
                    
                    # --- Section to display Image-by-Image Distance ---
                    with st.expander("📊 View Detailed Voting Results & Distances"):
                        st.write("**Vote Breakdown:**")
                        for name, count in vote_counts.items():
                            percentage = (count / len(all_preds)) * 100
                            st.write(f"- {name}: {count} images ({percentage:.1f}%)")
                            
                        st.write("---")
                        st.markdown("### 📏 Image-by-Image Distance Analysis:")
                        st.write("*Note: Distance must be below the Threshold to be classified.*")
                        
                        cols = st.columns(min(len(image_details), 5)) 
                        for i, detail in enumerate(image_details):
                            if i < 5:
                                img = Image.open(detail["file"])
                                cols[i].image(img, use_container_width=True)
                                
                                # Display the calculated distance prominently
                                cols[i].markdown(f"**Dist: {detail['dist']:.2f}**")
                                
                                # Indicate Pass or Fail based on the OOD status
                                if "Unknown" in detail["pred"]:
                                    cols[i].error("OOD (Unknown)")
                                else:
                                    cols[i].success(detail["pred"].split(".")[0]) # Display only the genus name

                        if len(uploaded_files) > 5:
                            st.write(f"... and {len(uploaded_files) - 5} more images processed.")

# ---------------------------------------------------------
# TAB 2: COLONY COUNTING
# ---------------------------------------------------------
with tab2:
    st.header("🧮 Edge-Based Bacterial Colony Counting")
    col1, col2 = st.columns(2)
    with col1:
        min_area = st.slider("Minimum Colony Size (Pixels)", 1, 100, 10)
    with col2:
        max_area = st.slider("Maximum Colony Size (Pixels)", 500, 10000, 5000)

    count_file = st.file_uploader("📤 Upload Agar Plate Image", type=["jpg","png","jpeg"], key="count")

    if count_file is not None:
        image = Image.open(count_file).convert('RGB')
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colony_count = 0
        img_with_boxes = img_array.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                colony_count += 1

        st.markdown("---")
        st.metric(label="Total Colonies Detected (CFUs)", value=colony_count)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with img_col2:
            st.image(img_with_boxes, caption="Detected Colonies", use_container_width=True)