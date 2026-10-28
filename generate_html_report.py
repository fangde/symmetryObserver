import os
import base64

def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

def update_full_report():
    img_grid = get_base64_img("results/transformer_grid_visualization.png")
    img_spring = get_base64_img("spring_benchmark/results/spring_phase_portrait_comparison.png")
    img_pairs = get_base64_img("spring_benchmark/results/phase_correspondence_10pairs.png")
    img_tendon = get_base64_img("tendon_benchmark/results/global_transformation_comparison.png")
    img_cascade = get_base64_img("tendon_benchmark/results/cascade_affine_lddmm_orthogonal.png")
    img_control = get_base64_img("tendon_benchmark/results/closed_loop_control_comparison.png")
    
    with open("template_report.html", "r", encoding="utf-8") as f:
        html = f.read()
        
    html = html.replace("__IMG_GRID__", img_grid)
    html = html.replace("__IMG_SPRING__", img_spring)
    html = html.replace("__IMG_PAIRS__", img_pairs)
    html = html.replace("__IMG_TENDON__", img_tendon)
    html = html.replace("__IMG_CASCADE__", img_cascade)
    html = html.replace("__IMG_CONTROL__", img_control)
    
    workspace_path = "/Volumes/KIOXIA/symmetryObserver/test_report.html"
    with open(workspace_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Updated workspace HTML report at: {workspace_path}")

    brain_path = "/Users/fangdeliu/.gemini/antigravity/brain/8988dc4f-935c-4c53-819e-72e1f2892381/test_report.html"
    with open(brain_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Updated brain HTML report at: {brain_path}")

if __name__ == "__main__":
    update_full_report()
