import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
import cv2
import shutil
from collections import Counter

st.set_page_config(page_title="Pathogen Identification System", layout="wide")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# ==========================================
# ENTERPRISE-GRADE UI INJECTIONS
# ==========================================

def inject_login_ui():
    """CSS applied only for the login screen with Microscopy Background"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    /* High-end Medical/Microbiology Background */
    .stApp {
        background-color: #0d1117;
        /* Using a high-quality abstract microbiology/cell image with a dark gradient overlay */
        background-image: 
            linear-gradient(to bottom, rgba(13, 17, 23, 0.8), rgba(13, 17, 23, 0.95)),
            url('https://images.unsplash.com/photo-1614935151651-0bea6508ad6b?q=80&w=2000&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Hide sidebar and top header on login */
    [data-testid="collapsedControl"] { display: none !important; }
    header { display: none !important; }

    /* Glassmorphism Login Card */
    div[data-testid="stForm"] {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
    }

    /* Typography */
    h2 { color: #ffffff !important; font-weight: 600 !important; letter-spacing: -0.5px; }
    p { color: #8b949e !important; }

    /* Form Inputs */
    .stTextInput input {
        background-color: rgba(13, 17, 23, 0.7) !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #c9d1d9 !important;
        padding: 10px 14px;
    }
    .stTextInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 1px #58a6ff !important;
    }

    /* Primary Button */
    .stButton > button {
        background-color: #238636;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { background-color: #2ea043; border-color: rgba(240, 246, 252, 0.1); }
    
    /* Error Messages */
    .stAlert { border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.1); }
    </style>
    """, unsafe_allow_html=True)

def inject_main_ui():
    """CSS applied for the main dashboard after login"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    .stApp { background-color: #0d1117; color: #c9d1d9; background-image: none; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 500 !important; letter-spacing: -0.5px; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }

    div.stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #30363d; gap: 24px; }
    div.stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 500; border: none !important; background: transparent !important; padding-bottom: 12px; padding-top: 12px; }
    div.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

    .stButton > button { background-color: #21262d; color: #c9d1d9 !important; border: 1px solid #30363d; border-radius: 6px; padding: 6px 16px; font-weight: 500; transition: all 0.2s ease; width: 100%; }
    .stButton > button:hover { background-color: #30363d; border-color: #8b949e; }
    .stButton > button:active { background-color: #282e33; }

    .stTextInput input, .stSelectbox div[data-baseweb="select"] { background-color: #0d1117 !important; border: 1px solid #30363d !important; border-radius: 6px !important; color: #c9d1d9 !important; }
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within { border-color: #58a6ff !important; box-shadow: 0 0 0 1px #58a6ff !important; }

    .stFileUploader > div > div { background-color: #161b22 !important; border: 1px dashed #30363d !important; border-radius: 6px !important; }
    div[data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }
    .stAlert { border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.1); }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 2.2rem !important; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# AUTHENTICATION ROUTING
# ==========================================

if not st.session_state['authenticated']:
    inject_login_ui()
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>Secure Access Gateway</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Automated Light Microscopy Pathogen System</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate")

            if submit:
                # Default Credentials
                if username == "admin" and password == "admin123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("Authentication Failed: Invalid credentials.")
    
    st.stop() # Stops the rest of the code from running if not logged in

# If authenticated, apply the main dashboard styling
inject_main_ui()

# ==========================================
# MAIN APPLICATION
# ==========================================

st.title("Automated Light Microscopy Pathogen System")
st.markdown("Integrated framework for **Pathogen Identification (ProtoNet)** and **Edge-Based Colony Counting**.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "advanced_protonet_resnet18.pth"
SUPPORT_DIR = "Inbuilt_Support"
NUM_SUPPORT_PER_CLASS = 5

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

class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

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
        st.error(f"Weights load error (Ignore if testing UI only): {e}")
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

model = load_model(MODEL_PATH)
prototypes = build_prototypes(SUPPORT_DIR, model, NUM_SUPPORT_PER_CLASS)


# ==========================================
# Sidebar Settings
# ==========================================
st.sidebar.header("System Settings")
THRESHOLD = st.sidebar.slider(
    "OOD Detection Threshold", 
    min_value=0.0, 
    max_value=80.0, 
    value=38.0, 
    step=1.0, 
    help="Lower values are stricter. If a sample's distance is higher than this, it is flagged as Unknown."
)

st.sidebar.markdown("---")
st.sidebar.header("Support Set Manager")

os.makedirs(SUPPORT_DIR, exist_ok=True)
existing_classes = sorted([d for d in os.listdir(SUPPORT_DIR) if os.path.isdir(os.path.join(SUPPORT_DIR, d))])

if not existing_classes:
    st.sidebar.info("No custom classes added to memory yet.")
else:
    with st.sidebar.expander(f"View / Delete Classes ({len(existing_classes)} total)"):
        for cls in existing_classes:
            st.write(f"- {cls}")
        
        st.markdown("---")
        class_to_delete = st.selectbox("Select a class to remove:", existing_classes)
        if st.button("Delete Selected Class"):
            dir_to_remove = os.path.join(SUPPORT_DIR, class_to_delete)
            try:
                shutil.rmtree(dir_to_remove) 
                st.cache_resource.clear()    
                st.success(f"'{class_to_delete}' deleted successfully.")
                st.rerun()                   
            except Exception as e:
                st.error(f"Error deleting folder: {e}")

st.sidebar.markdown("---")

st.sidebar.subheader("Add New Species")
new_class_name = st.sidebar.text_input("New Species Name:")
new_class_files = st.sidebar.file_uploader("Upload exactly 5 images (5-Shot)", accept_multiple_files=True, type=["jpg","jpeg","png"])

if st.sidebar.button("Update Prototype Memory"):
    if len(new_class_files) != 5:
        st.sidebar.error("Please upload exactly 5 images.")
    elif not new_class_name:
        st.sidebar.error("Please enter a class name.")
    else:
        new_dir = os.path.join(SUPPORT_DIR, new_class_name)
        os.makedirs(new_dir, exist_ok=True)
        for f in new_class_files:
            with open(os.path.join(new_dir, f.name), "wb") as out_file:
                out_file.write(f.getbuffer())
        st.cache_resource.clear()
        st.sidebar.success(f"'{new_class_name}' added. Refreshing...")
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Logout Session"):
    st.session_state['authenticated'] = False
    st.rerun()

# ==========================================
# Main Workspace
# ==========================================
tab1, tab2 = st.tabs(["Pathogen Identification (Few-Shot)", "Edge-Based Colony Counting"])

with tab1:
    if len(prototypes) == 0:
        st.warning("No class prototypes found in memory. Please use the sidebar to add a new support set.")
    else:
        st.success(f"System Online: {len(prototypes)} distinct bacterial species loaded into Prototype Memory.")

        uploaded_files = st.file_uploader("Upload microscopy images for analysis (Minimum 5 required)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True, key="query")

        if uploaded_files:
            if len(uploaded_files) < 5:
                st.warning(f"You have uploaded {len(uploaded_files)} image(s). Please upload at least 5 images for Majority Voting.")
            else:
                st.info("Processing images via ResNet18. Please wait...")
                
                all_preds = []
                image_details = [] 
                
                for uploaded_file in uploaded_files:
                    try:
                        test_img = Image.open(uploaded_file).convert("RGB")
                        test_emb = image_to_embedding(test_img, model).numpy()

                        class_names = list(prototypes.keys())
                        proto_matrix = np.stack([prototypes[c] for c in class_names], axis=0)
                        
                        dists = np.linalg.norm(proto_matrix - test_emb[None, :], axis=1)
                        min_idx = int(np.argmin(dists))
                        min_dist = float(dists[min_idx]) 
                        
                        if min_dist > THRESHOLD:
                            pred_label = "Unknown Species (Out of Distribution)"
                        else:
                            pred_label = class_names[min_idx]
                            
                        all_preds.append(pred_label)
                        
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
                        st.error(f"ALERT: High-distance anomaly detected.\n\nFinal Prediction: {best_answer}")
                    else:
                        st.success(f"Final Predicted Bacteria: {best_answer}")
                        st.write(f"*(Confidence: {max_votes} out of {len(all_preds)} images agreed on this result)*")
                        
                        st.markdown("### Clinical Information")
                        info = BACTERIA_INFO.get(best_answer)
                        if info:
                            st.info(f"**Description:** {info['description']}")
                            st.warning(f"**Associated Diseases:** {info['diseases']}")
                            st.success(f"**Clinical Details:** {info['details']}")
                    
                    with st.expander("View Detailed Voting Results & Distances"):
                        st.write("**Vote Breakdown:**")
                        for name, count in vote_counts.items():
                            percentage = (count / len(all_preds)) * 100
                            st.write(f"- {name}: {count} images ({percentage:.1f}%)")
                            
                        st.write("---")
                        st.markdown("### Image-by-Image Distance Analysis:")
                        st.write("*Note: Distance must be below the Threshold to be classified.*")
                        
                        cols = st.columns(min(len(image_details), 5)) 
                        for i, detail in enumerate(image_details):
                            if i < 5:
                                img = Image.open(detail["file"])
                                cols[i].image(img, use_container_width=True)
                                
                                cols[i].markdown(f"**Dist: {detail['dist']:.2f}**")
                                
                                if "Unknown" in detail["pred"]:
                                    cols[i].error("OOD (Unknown)")
                                else:
                                    cols[i].success(detail["pred"].split(".")[0]) 

                        if len(uploaded_files) > 5:
                            st.write(f"... and {len(uploaded_files) - 5} more images processed.")

with tab2:
    st.header("Edge-Based Bacterial Colony Counting")
    col1, col2 = st.columns(2)
    with col1:
        min_area = st.slider("Minimum Colony Size (Pixels)", 1, 100, 10)
    with col2:
        max_area = st.slider("Maximum Colony Size (Pixels)", 500, 10000, 5000)

    count_file = st.file_uploader("Upload Agar Plate Image", type=["jpg","png","jpeg"], key="count")

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