import os
from PIL import Image

# ඔයාගේ ලොකු ෆෝල්ඩරේ නම
source_dir = "FSL_Dataset/all_classes" 
# හැදෙන්න ඕන අලුත් පොඩි ෆෝල්ඩරේ නම
target_dir = "Inbuilt_Support"

os.makedirs(target_dir, exist_ok=True)

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

print("⏳ පින්තූර Compress වෙනවා... කරුණාකර රැඳී සිටින්න.")

for cls in classes:
    source_class_dir = os.path.join(source_dir, cls)
    target_class_dir = os.path.join(target_dir, cls)
    os.makedirs(target_class_dir, exist_ok=True)
    
    # පින්තූර 5ක් විතරක් තෝරගන්නවා (tif, tiff ඔක්කොම ගන්නවා)
    files = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    selected_files = files[:5] 
    
    for file in selected_files:
        src_path = os.path.join(source_class_dir, file)
        
        # අලුත් නම හදනවා .jpg වලින් (size එක අඩු වෙන්න)
        filename_without_ext = os.path.splitext(file)[0]
        dst_path = os.path.join(target_class_dir, f"{filename_without_ext}.jpg")
        
        try:
            # පින්තූරෙ ඕපන් කරලා, 512x512 වලට වඩා ලොකු නම් පොඩි කරලා, JPG විදිහට සේව් කරනවා
            img = Image.open(src_path).convert("RGB")
            img.thumbnail((512, 512)) # Quality එක බහින්නෙ නෑ, සයිස් එක විතරයි අඩු වෙන්නේ
            img.save(dst_path, "JPEG", quality=85)
        except Exception as e:
            print(f"⚠️ Error with {file}: {e}")

print(f"✅ වැඩේ ඉවරයි! දැන් අලුත් '{target_dir}' ෆෝල්ඩරේ සයිස් එක Properties ගිහින් බලන්න.")