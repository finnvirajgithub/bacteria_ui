import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
from collections import Counter

st.set_page_config(page_title="Bacteria FSL", layout="wide")

# --- UI Header ---
st.title("🦠 Few-Shot Classification (ProtoNet)")
st.markdown("Automated Light Microscopy Pathogen Identification System")

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "advanced_protonet_resnet18.pth"
SUPPORT_DIR = "Inbuilt_Support" # අලුත් කුඩා ෆෝල්ඩරය
NUM_SUPPORT_PER_CLASS = 5
THRESHOLD = 45.0  # Open-Set Recognition Threshold

# ==========================================
# BACTERIA KNOWLEDGE BASE (Clinical Data)
# ==========================================
BACTERIA_INFO = {
    "Acinetobacter.baumanii": {
        "description": "Gram-negative, strictly aerobic, pleomorphic coccobacillus.",
        "diseases": "Hospital-acquired pneumonia, Bloodstream infections, Meningitis, Wound infections.",
        "details": "Highly associated with hospital-acquired (nosocomial) infections. Known for severe multi-drug resistance (CRAB)."
    },
    "Bacteroides.fragilis": {
        "description": "Gram-negative, obligately anaerobic, rod-shaped bacterium.",
        "diseases": "Intra-abdominal infections, Pelvic inflammatory disease, Bacteremia.",
        "details": "Part of the normal human colon flora but becomes highly pathogenic if displaced into the bloodstream or surrounding tissue."
    },
    "Enterococcus.faecalis": {
        "description": "Gram-positive, commensal bacterium inhabiting the gastrointestinal tracts.",
        "diseases": "Urinary Tract Infections (UTIs), Endocarditis, Bacteremia, Wound infections.",
        "details": "Can cause life-threatening infections in humans, especially in hospital environments. High natural resistance to many antibiotics."
    },
    "Escherichia.coli": {
        "description": "Gram-negative, facultatively anaerobic, rod-shaped coliform bacterium.",
        "diseases": "Urinary Tract Infections (UTIs), Gastroenteritis, Hemorrhagic colitis.",
        "details": "While mostly harmless in the gut, pathogenic strains (like O157:H7) can cause severe foodborne illnesses and kidney failure."
    },
    "Listeria.monocytogenes": {
        "description": "Gram-positive, rod-shaped bacterium capable of surviving in the presence or absence of oxygen.",
        "diseases": "Listeriosis, Meningitis, Encephalitis, Severe infections in newborns.",
        "details": "A dangerous foodborne pathogen that can grow and multiply even at refrigerator temperatures (4°C)."
    },
    "Neisseria.gonorrhoeae": {
        "description": "Gram-negative diplococci (occurring in pairs) bacteria.",
        "diseases": "Gonorrhea, Pelvic Inflammatory Disease (PID), Septic arthritis.",
        "details": "A sexually transmitted pathogen. It is highly adaptive and currently showing increasing resistance to multiple antibiotic classes."
    },
    "Proteus": {
        "description": "Gram-negative, facultatively anaerobic, rod-shaped bacterium with swarming motility.",
        "diseases": "Complicated Urinary Tract Infections (UTIs), Kidney stones (Struvite stones).",
        "details": "Produces the enzyme urease, which makes urine alkaline and leads to the rapid formation of kidney stones."
    },
    "Pseudomonas.aeruginosa": {
        "description": "Gram-negative, aerobic, rod-shaped bacterium with unipolar motility.",
        "diseases": "Ventilator-associated pneumonia, Sepsis, Infections in burn wounds and cystic fibrosis.",
        "details": "Known for its distinctive green-blue pigment (pyocyanin) and a fruity odor. Highly resistant to a wide range of antibiotics."
    },
    "Staphylococcus.aureus": {
        "description": "Gram-positive, round-shaped bacterium frequently found in the upper respiratory tract and on the skin.",
        "diseases": "Skin and soft tissue infections, Pneumonia, Endocarditis, Osteomyelitis.",
        "details": "Forms golden-yellow colonies. Methicillin-resistant strains (MRSA) are a major global healthcare concern."
    },
    "Staphylococcus.epidermidis": {
        "description": "Gram-positive bacterium, and one of the most common species of the normal human skin flora.",
        "diseases": "Infections of intravascular devices (catheters, prosthetics), Endocarditis.",
        "details": "Not typically pathogenic, but forms thick biofilms on plastic implants and catheters, making infections very hard to treat."
    },
    "Streptococcus.agalactiae": {
        "description": "Gram-positive coccus that forms chains. Also known as Group B Streptococcus (GBS).",
        "diseases": "Neonatal meningitis, Pneumonia, Sepsis in newborns.",
        "details": "A common asymptomatic colonizer of the gastrointestinal and genital tracts, but highly dangerous to infants during childbirth."
    },
    "Streptococcus.pneumoniae": {
        "description": "Gram-positive, lancet-shaped diplococci bacterium.",
        "diseases": "Community-acquired pneumonia, Otitis media (ear infections), Sinusitis, Meningitis.",
        "details": "Possesses a thick polysaccharide capsule that makes it resistant to phagocytosis (being eaten by immune cells)."
    },
    "Candida.albicans": {
        "description": "An opportunistic pathogenic yeast (fungus), not a typical bacterium.",
        "diseases": "Oral thrush, Vaginal yeast infections, Invasive candidiasis.",
        "details": "Usually lives harmlessly in the human body, but can overgrow and cause severe infections in immunocompromised individuals."
    }
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
    # Cloud එකේ Memory අවුල් නොඑන්න CPU එකට Load කරනවා
    m = ProtoNet().to(torch.device("cpu"))
    try:
        state = torch.load(model_path, map_location=torch.device("cpu"))
        m.load_state_dict(state)
    except Exception as e:
        st.error(f"Weights ලෝඩ් වුණේ නෑ!: {e}")
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
    
    # Folder එක නැත්නම් ඔටෝම හදනවා (Error එන්නේ නෑ)
    os.makedirs(support_dir, exist_ok=True)
        
    classes = sorted([d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))])
    
    # Folder එක ඇතුළේ බැක්ටීරියා මුකුත් නැත්නම්, හිස් මතකය Return කරනවා
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
# Sidebar: Support Set Updater (අලුත් බැක්ටීරියා දාන තැන)
# ==========================================
st.sidebar.header("⚙️ Support Set Manager")
st.sidebar.write("Add a novel bacterial class to the memory.")

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
                
        # Cache එක clear කරලා App එක Refresh කරනවා අලුත් මතකය ගන්න
        st.cache_resource.clear()
        st.sidebar.success(f"✅ '{new_class_name}' added successfully! Page will refresh...")
        st.rerun()

# ==========================================
# MAIN UI: Image Analysis (Query Images)
# ==========================================
if len(prototypes) == 0:
    st.warning("No class prototypes found in memory. Please use the sidebar to add a new support set.")
    st.stop()

st.success(f"✅ System Online: {len(prototypes)} distinct bacterial species loaded into Prototype Memory.")

uploaded_files = st.file_uploader("📤 Upload microscopy images for analysis (Minimum 5 images required)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 5:
        st.warning(f"⚠️ You have only uploaded {len(uploaded_files)} image(s). Please upload at least 5 images to run the analysis (Majority Voting).")
    else:
        st.info(f"⏳ Processing {len(uploaded_files)} images via ResNet18... Please wait.")
        all_preds = []
        
        for uploaded_file in uploaded_files:
            try:
                test_img = Image.open(uploaded_file).convert("RGB")
                test_emb = image_to_embedding(test_img, model).numpy()

                class_names = list(prototypes.keys())
                proto_matrix = np.stack([prototypes[c] for c in class_names], axis=0)
                
                dists = np.linalg.norm(proto_matrix - test_emb[None, :], axis=1)
                min_idx = int(np.argmin(dists))
                min_dist = dists[min_idx]
                
                if min_dist > THRESHOLD:
                    all_preds.append("Unknown Species (Out of Distribution)")
                else:
                    all_preds.append(class_names[min_idx])
                    
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
                
                # --- INFO DISPLAY SECTION ---
                st.markdown("### 📋 Clinical Information")
                
                info = BACTERIA_INFO.get(best_answer)
                
                if info:
                    st.info(f"**🔬 Description:** {info['description']}")
                    st.warning(f"**⚠️ Associated Diseases:** {info['diseases']}")
                    st.success(f"**💡 Clinical Details:** {info['details']}")
                else:
                    st.info("⚠️ No clinical information available in the knowledge base for this newly added species.")
            
            with st.expander("📊 View Detailed Voting Results"):
                st.write("Breakdown according to processed images:")
                for name, count in vote_counts.items():
                    percentage = (count / len(all_preds)) * 100
                    st.write(f"- **{name}**: {count} images ({percentage:.1f}%)")
                    
                st.write("---")
                cols = st.columns(min(len(uploaded_files), 5)) 
                for i, col in enumerate(cols):
                    if i < 5:
                        img = Image.open(uploaded_files[i])
                        col.image(img, use_container_width=True) 
                if len(uploaded_files) > 5:
                    st.write(f"... and {len(uploaded_files) - 5} more images processed.")