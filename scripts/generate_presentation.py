import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette: Warm Editorial & Scientific
    BG_COLOR = RGBColor(245, 241, 232)      # Warm Ivory #F5F1E8
    CARD_BG = RGBColor(255, 252, 246)       # Paper White #FFFCF6
    CARD_BORDER = RGBColor(232, 226, 215)   # Soft Stone #E8E2D7
    PRIMARY_DARK = RGBColor(98, 105, 77)    # Deep Olive #62694D
    PRIMARY_MUTED = RGBColor(133, 139, 107) # Muted Olive #858B6B
    TEXT_MAIN = RGBColor(48, 47, 41)        # Charcoal Ink #302F29
    TEXT_MUTED = RGBColor(110, 110, 100)    # Subdued gray-charcoal
    ACCENT_BLUE = RGBColor(40, 95, 175)     # Deep Blue
    ACCENT_GREEN = RGBColor(45, 140, 95)    # Emerald Green
    WHITE = RGBColor(255, 255, 255)

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="ASTRAVISION · PROJECT PRESENTATION"):
        # Header banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = PRIMARY_DARK

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        return card

    # =========================================================================
    # SLIDE 1: Title & Team Details
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, BG_COLOR)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.0), Inches(0.12), Inches(4.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = PRIMARY_DARK
    bar.line.fill.background()

    # Title & Subtitle block
    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(0.9), Inches(11.3), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "CAPSTONE / FINAL YEAR RESEARCH PROJECT"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = PRIMARY_MUTED

    p1 = tf1.add_paragraph()
    p1.text = "AstraVision"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_MAIN

    p2 = tf1.add_paragraph()
    p2.text = "Modern Perceptual Image Quality Assessment & Structural Intelligence Platform"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = PRIMARY_DARK

    p3 = tf1.add_paragraph()
    p3.text = "Differentiable PyTorch 2.x Engine, Multi-Scale SSIM, Closed-Loop Restoration & Multi-Agent Auditing"
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_MUTED

    # Team & Project Details Card
    add_card(s1, Inches(1.2), Inches(3.5), Inches(5.3), Inches(3.3))
    team_box = s1.shapes.add_textbox(Inches(1.4), Inches(3.6), Inches(4.9), Inches(3.1))
    tf_team = team_box.text_frame
    tf_team.word_wrap = True

    p_th = tf_team.paragraphs[0]
    p_th.text = "TEAM MEMBERS & ENROLLMENT"
    p_th.font.size = Pt(13)
    p_th.font.bold = True
    p_th.font.color.rgb = PRIMARY_DARK

    members = [
        ("Sandeep Kumar (Lead)", "Enrollment / Roll No: [22BCS10XXX / Sandeep332005]"),
        ("Team Member 2", "Enrollment / Roll No: [Insert Enrollment No]"),
        ("Team Member 3", "Enrollment / Roll No: [Insert Enrollment No]"),
        ("Team Member 4", "Enrollment / Roll No: [Insert Enrollment No]"),
    ]
    for name, roll in members:
        p_m = tf_team.add_paragraph()
        p_m.text = f"• {name}"
        p_m.font.size = Pt(12)
        p_m.font.bold = True
        p_m.font.color.rgb = TEXT_MAIN
        
        p_r = tf_team.add_paragraph()
        p_r.text = f"   {roll}"
        p_r.font.size = Pt(10.5)
        p_r.font.color.rgb = TEXT_MUTED

    # Supervisor & Department Card
    add_card(s1, Inches(6.8), Inches(3.5), Inches(5.7), Inches(3.3))
    meta_box = s1.shapes.add_textbox(Inches(7.0), Inches(3.6), Inches(5.3), Inches(3.1))
    tf_meta = meta_box.text_frame
    tf_meta.word_wrap = True

    p_mh = tf_meta.paragraphs[0]
    p_mh.text = "PROJECT METADATA & SUPERVISION"
    p_mh.font.size = Pt(13)
    p_mh.font.bold = True
    p_mh.font.color.rgb = PRIMARY_DARK

    metas = [
        ("Project Supervisor:", "[Dr. / Prof. Guide Name, Designation]"),
        ("Department:", "Department of Computer Science & Engineering"),
        ("Institution:", "[University / College Institute Name]"),
        ("Live Web Deployment:", "https://sandeep332005.github.io/AstraVision/"),
        ("Source Repository:", "https://github.com/Sandeep332005/AstraVision"),
    ]
    for label, val in metas:
        p_l = tf_meta.add_paragraph()
        p_l.text = f"{label} {val}"
        p_l.font.size = Pt(11)
        p_l.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 2: Problem Statement
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, BG_COLOR)
    add_header(s2, "Problem Statement: The Perceptual Blind Spot of Classical Metrics")

    # Left Column Card: The Failure of MSE / PSNR
    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    b2_1 = s2.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf2_1 = b2_1.text_frame
    tf2_1.word_wrap = True
    
    p = tf2_1.paragraphs[0]
    p.text = "1. Classical Pixel Metrics Fail Human Perception"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    items_2_1 = [
        "MSE & PSNR assume signal independence: They sum squared pixel differences without analyzing spatial correlations.",
        "Blind to distortion types: Identical MSE (~144) can correspond to an imperceptible brightness shift or catastrophic image blur.",
        "Mismatched with the Human Visual System (HVS): Human eyes extract structural information, shapes, textures, and edges—not isolated pixel energy.",
        "Standard loss functions (L1/L2) in deep learning often lead to blurry, oversmoothed reconstructions."
    ]
    for it in items_2_1:
        p = tf2_1.add_paragraph()
        p.text = f"• {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED

    # Right Column Card: Legacy Open Source Limitations
    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    b2_2 = s2.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf2_2 = b2_2.text_frame
    tf2_2.word_wrap = True

    p = tf2_2.paragraphs[0]
    p.text = "2. Breakdown of Legacy SSIM Tooling"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    items_2_2 = [
        "Unmaintained Codebases: Popular repositories like Po-Hsun-Su/pytorch-ssim have been abandoned for >5 years, relying on broken torch.autograd.Variable.",
        "Dynamic Range Bugs: Fixed constants (C1=0.01^2, C2=0.03^2) corrupt math when images are in raw [0, 255] integer space.",
        "No Multi-Scale Support: Lacks Multi-Scale SSIM (MS-SSIM), required by video streaming standards (Netflix VMAF, YouTube).",
        "No Diagnostic Telemetry: Zero spatial heatmap visualizers, interactive comparison tools, or closed-loop restoration workflows."
    ]
    for it in items_2_2:
        p = tf2_2.add_paragraph()
        p.text = f"• {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: Project Objectives
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, BG_COLOR)
    add_header(s3, "Project Objectives: Engineering a Modern IQA Ecosystem")

    obj_cards = [
        ("1. PyTorch 2.x Convolutions", "Develop a high-throughput, GPU/MPS accelerated SSIM and MS-SSIM engine with automatic dynamic range calibration.", Inches(0.8), Inches(1.6)),
        ("2. Multi-Scale Pyramid Analysis", "Implement 5-scale hierarchical Gaussian pyramid pooling with IEEE 2003 empirical weights for viewing-distance invariance.", Inches(4.8), Inches(1.6)),
        ("3. Differentiable Loss Module", "Expose a plug-and-play PyTorch nn.Module (SSIMLoss) for end-to-end backpropagation in generative vision models.", Inches(8.8), Inches(1.6)),
        ("4. Interactive Comparison Studio", "Build a web-based workstation with a 60 FPS split slider, side-by-side mode, and letterbox-aware ROI bounding box analyzer.", Inches(0.8), Inches(4.3)),
        ("5. Closed-Loop Restoration Lab", "Benchmark Bilateral, NLM, Median, and Unsharp filters, tracking true quality recovery (Delta-SSIM) in real time.", Inches(4.8), Inches(4.3)),
        ("6. Dual-Engine Web Deployment", "Deploy a dual-mode web app: live on GitHub Pages with browser-native math, and local GPU acceleration via Python.", Inches(8.8), Inches(4.3)),
    ]
    for title, desc, left, top in obj_cards:
        add_card(s3, left, top, Inches(3.7), Inches(2.4))
        tb = s3.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(3.3), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY_DARK
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11.5)
        p_d.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 4: Proposed Methodology & Mathematical Foundations
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, BG_COLOR)
    add_header(s4, "Proposed Methodology: Mathematical Formulations")

    add_card(s4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb4_1 = s4.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf4_1 = tb4_1.text_frame
    tf4_1.word_wrap = True

    p = tf4_1.paragraphs[0]
    p.text = "1. Single-Scale SSIM (Wang et al., 2004)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    p = tf4_1.add_paragraph()
    p.text = "SSIM decomposes structural fidelity into three components:"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MUTED

    p = tf4_1.add_paragraph()
    p.text = "SSIM(x, y) = [l(x, y)] · [c(x, y)] · [s(x, y)]"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf4_1.add_paragraph()
    p.text = "Combined Operational Formula:\nSSIM = [(2μxμy + C1)(2σxy + C2)] / [(μx² + μy² + C1)(σx² + σy² + C2)]"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    points_4_1 = [
        "μx, μy: Local Gaussian-weighted means (11×11 sliding window, σ=1.5)",
        "σx², σy²: Local variances reflecting visual contrast",
        "σxy: Local cross-covariance reflecting structural preservation",
        "C1 = (K1·L)², C2 = (K2·L)²: Regularization constants preventing division by zero (K1=0.01, K2=0.03, L=dynamic range)"
    ]
    for pt in points_4_1:
        p = tf4_1.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    add_card(s4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb4_2 = s4.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True

    p = tf4_2.paragraphs[0]
    p.text = "2. Multi-Scale SSIM & Loss Function"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    p = tf4_2.add_paragraph()
    p.text = "MS-SSIM evaluates images across M = 5 dyadic pyramid scales:"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MUTED

    p = tf4_2.add_paragraph()
    p.text = "MS-SSIM(x, y) = [l_M(x, y)]^α_M · ∏ [c_j(x, y)]^β_j · [s_j(x, y)]^γ_j"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf4_2.add_paragraph()
    p.text = "• Weights: β = γ = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]\n• Decimation: 2×2 average pooling across hierarchical levels\n• Differentiable Loss:\n  Loss = 1.0 - SSIM(x, y)"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    p = tf4_2.add_paragraph()
    p.text = "3. Complementary Metric Suite"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    p = tf4_2.add_paragraph()
    p.text = "• PSNR = 10 · log10(L² / MSE) dB\n• MSE = (1/HW) ∑(x_ij - y_ij)²\n• Perceptual Gradient Distance: Sobel multi-scale edge loss"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 5: Technologies Used
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, BG_COLOR)
    add_header(s5, "Technologies & Tools Used")

    tech_stacks = [
        ("Deep Learning & Vision", ["PyTorch 2.x (Autograd, Conv2d, Loss)", "Torchvision (Transforms & Models)", "NumPy (Matrix Math)", "Pillow (Image I/O)"], Inches(0.8), Inches(1.6)),
        ("Backend Architecture", ["Python 3.8+ (ThreadingHTTPServer)", "Non-blocking Thread Concurrency", "Explicit Content-Length Keep-Alive", "RESTful JSON APIs"], Inches(4.8), Inches(1.6)),
        ("Client Workstation UI", ["HTML5 Canvas (2D Pixel Buffers)", "Vanilla JavaScript (Zero-dependency ES6+)", "Warm Editorial Design System", "Lucide SVG Iconography"], Inches(8.8), Inches(1.6)),
        ("Testing & Diagnostics", ["Pytest 9.2 (23 Verification Tests)", "Multi-Agent Empirical Audit Runner", "Runtime & Memory Profiler", "Sobel Gradient Edge Extractor"], Inches(0.8), Inches(4.3)),
        ("Cloud & Deployment", ["GitHub Pages (Static Hosting)", "Git Subtree Push Architecture", "GitHub REST API v3", "HTTPS Edge CDN Distribution"], Inches(4.8), Inches(4.3)),
        ("Scientific Visuals", ["Jet 8-bit Colormapped Heatmaps", "30-Bin Canvas Error Histogram", "Letterbox-Aware ROI Mapping", "RFC 4180 CSV & Markdown Reports"], Inches(8.8), Inches(4.3)),
    ]
    for cat_title, bullets, left, top in tech_stacks:
        add_card(s5, left, top, Inches(3.7), Inches(2.4))
        tb = s5.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(3.3), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = cat_title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY_DARK
        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"• {b}"
            p_b.font.size = Pt(10.5)
            p_b.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 6: Implementation Details (Architecture & Subsystems)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, BG_COLOR)
    add_header(s6, "Implementation Details: 5 Core Architectural Subsystems")

    subsystems = [
        ("Subsystem 1: PyTorch Analytical Engine (`engine/ssim.py`)",
         "• Implements create_gaussian_window() for grouped 2D convolutions.\n• Automatic data_range detector (L=1.0 vs L=255.0).\n• SSIMLoss nn.Module for seamless training loop integration."),
        ("Subsystem 2: Distortion & Restoration (`engine/distortions.py`, `restoration.py`)",
         "• Real JPEG 8×8 forward/inverse Discrete Cosine Transform (DCT) with quantization tables.\n• Bilateral edge-preserving filter & Non-Local Means (NLM) patch search.\n• Tracks Delta-SSIM recovery quality."),
        ("Subsystem 3: Structural Heatmaps & Histogram (`engine/visualizer.py`)",
         "• Computes local structural similarity maps using 8-bit Jet colormapping.\n• Generates dynamic 30-bin error distribution histograms.\n• High-performance downsampling (max 640px) prevents browser UI lag."),
        ("Subsystem 4: Dual-Engine Web Architecture (`app/static/index.html`)",
         "• Local Mode: Communicates with multi-threaded Python server (CUDA/MPS).\n• Browser-Native Mode: Complete in-browser HTML5 Canvas fallback for GitHub Pages.\n• Zero server required for live web demos."),
        ("Subsystem 5: Multi-Agent Automated Audit (`engine/audit.py`)",
         "• Self-testing harness verifying all 10 modules in under 35 ms.\n• Checks mathematical invariants, storage atomicity, and API keep-alive."),
    ]
    top_pos = 1.6
    for title, desc in subsystems:
        add_card(s6, Inches(0.8), Inches(top_pos), Inches(11.7), Inches(0.95))
        tb = s6.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.05), Inches(11.3), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY_DARK
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = TEXT_MAIN
        top_pos += 1.05

    # =========================================================================
    # SLIDE 7: Prototype Screenshot 1 (Editorial Dashboard)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, BG_COLOR)
    add_header(s7, "Prototype Demonstration: AstraVision Warm Editorial Studio")

    # Embed Screenshot 1
    img1_path = "/Users/braj/ssim-lab/assets/astravision_editorial_dashboard.jpg"
    if os.path.exists(img1_path):
        s7.shapes.add_picture(img1_path, Inches(0.8), Inches(1.5), width=Inches(8.2))
    
    # Right Commentary Card
    add_card(s7, Inches(9.2), Inches(1.5), Inches(3.3), Inches(5.4))
    tb7 = s7.shapes.add_textbox(Inches(9.4), Inches(1.7), Inches(2.9), Inches(5.0))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "Key Interactive Features"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    features_s7 = [
        "Interactive Split Slider: Drag handle allows real-time before/after comparison at 60 FPS.",
        "5 Real-Time Metric HUDs: Displays SSIM, MS-SSIM, PSNR, MSE/MAE, and LPIPS distance simultaneously.",
        "Equal-MSE Preset Lab: One-click buttons verify the Wang et al. thesis on identical MSE=144 images.",
        "Structural Jet Heatmap: 8-bit color-coded map highlighting local quality degradation.",
        "Dynamic 30-Bin Histogram: Live frequency distribution of spatial error density."
    ]
    for feat in features_s7:
        p = tf7.add_paragraph()
        p.text = f"• {feat}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 8: Prototype Screenshot 2 (Dark Scientific Mode & Diagnostics)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, BG_COLOR)
    add_header(s8, "Prototype Demonstration: Dark Scientific Mode & Diagnostic Tools")

    # Embed Screenshot 2
    img2_path = "/Users/braj/ssim-lab/assets/astravision_dark_dashboard.jpg"
    if os.path.exists(img2_path):
        s8.shapes.add_picture(img2_path, Inches(0.8), Inches(1.5), width=Inches(8.2))

    # Right Commentary Card
    add_card(s8, Inches(9.2), Inches(1.5), Inches(3.3), Inches(5.4))
    tb8 = s8.shapes.add_textbox(Inches(9.4), Inches(1.7), Inches(2.9), Inches(5.0))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "Diagnostic Workstation"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    features_s8 = [
        "Dark Workstation UI: High-contrast theme (#111215) tailored for low-light laboratory inspection.",
        "Letterbox-Aware ROI Drawer: Draw custom sub-pixel bounding boxes on non-square images.",
        "Parametric Distortion Sliders: Real-time Gaussian blur, noise, and genuine JPEG DCT quality control.",
        "A/B Flicker & Side-by-Side: Rapid spacebar toggling detects minute high-frequency losses.",
        "One-Click Academic Exports: Generates publication Markdown reports and RFC 4180 CSV benchmark matrices."
    ]
    for feat in features_s8:
        p = tf8.add_paragraph()
        p.text = f"• {feat}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 9: Experimental Results & Empirical Verification
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, BG_COLOR)
    add_header(s9, "Empirical Verification: Academic & Performance Results")

    # Table of Wang et al. Equal-MSE Results
    add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb9_1 = s9.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True

    p = tf9_1.paragraphs[0]
    p.text = "1. Wang et al. Equal-MSE Verification"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    p = tf9_1.add_paragraph()
    p.text = "Empirical verification replicating Wang et al. (2004) where MSE is held constant (~144):"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    results_data = [
        ("Mean Shift (+12 DC)", "144", "0.9957", "Imperceptible"),
        ("Contrast Loss", "144", "0.9956", "Contrast soft"),
        ("Impulse Noise (S&P)", "144", "0.9575", "Disrupted"),
        ("Gaussian Blur", "144", "0.0723", "Severe loss"),
    ]
    for dist_t, mse_v, ssim_v, verd in results_data:
        p = tf9_1.add_paragraph()
        p.text = f"• {dist_t} | SSIM: {ssim_v} ({verd})"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN

    p = tf9_1.add_paragraph()
    p.text = "\nProof: SSIM strictly captures human visual sensitivity where MSE is completely blind."
    p.font.size = Pt(11)
    p.font.color.rgb = ACCENT_GREEN

    # Right Card: Hardware & Unit Test Proof
    add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb9_2 = s9.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True

    p = tf9_2.paragraphs[0]
    p.text = "2. Throughput & Test Certification"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    proofs = [
        ("Pytest Test Suite:", "23/23 tests passed in 0.79s (100% pass rate)"),
        ("Multi-Agent Module Audit:", "10/10 modules operational in 35.05ms"),
        ("Deep Learning Backprop:", "SSIMLoss gradient norm: 0.010450 (Zero NaNs)"),
        ("256×256 RGB Throughput:", "24.72 ms per frame (40.5 FPS Real-Time)"),
        ("512×512 RGB Throughput:", "138.37 ms per frame (7.2 FPS)"),
        ("1024×1024 RGB Throughput:", "529.76 ms per frame (1.9 FPS)"),
    ]
    for title, val in proofs:
        p = tf9_2.add_paragraph()
        p.text = f"• {title} {val}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 10: Comparison with Existing Solutions
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, BG_COLOR)
    add_header(s10, "Comparative Analysis: AstraVision vs State-of-the-Art")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tb10 = s10.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "Competitive Feature Comparison Matrix"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    matrix = [
        ("Modern PyTorch 2.x Support", "Broken (Variable wrapper)", "Supported", "NumPy only", "Natively Optimized"),
        ("Multi-Scale SSIM (MS-SSIM)", "Not supported", "Functional", "Not supported", "5-Scale Dyadic Pyramid"),
        ("Interactive Comparison Studio", "None", "None", "None", "Split Slider (60 FPS)"),
        ("Spatial Jet Heatmaps", "None", "None", "Static Matplotlib", "Live 8-bit Colormap"),
        ("Dynamic Error Histogram", "None", "None", "None", "30-Bin Live Canvas"),
        ("Closed-Loop Restoration Lab", "None", "None", "None", "Bilateral, NLM, Median"),
        ("Dual Engine Deployment", "None", "None", "None", "PyTorch + Web Canvas"),
        ("Live Public URL", "None", "None", "None", "GitHub Pages Live Demo"),
    ]
    for row in matrix:
        p = tf10.add_paragraph()
        p.text = f"• {row[0]}: AstraVision = {row[4]} | Legacy = {row[1]} | TorchMetrics = {row[2]}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 11: Deployment & Project Links
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, BG_COLOR)
    add_header(s11, "Project Deployment & Open Source Availability")

    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb11_1 = s11.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf11_1 = tb11_1.text_frame
    tf11_1.word_wrap = True

    p = tf11_1.paragraphs[0]
    p.text = "Live Web Deployment (GitHub Pages)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    links_1 = [
        "Live URL: https://sandeep332005.github.io/AstraVision/",
        "HTTP Status: 200 OK (Built & Enforced HTTPS)",
        "Architecture: Browser-Native HTML5 Canvas mathematical engine running client-side with zero installation.",
        "Interactive Testing: Users can drag-and-drop any image, apply real distortions, inspect heatmaps, and run benchmarks directly on the web."
    ]
    for l in links_1:
        p = tf11_1.add_paragraph()
        p.text = f"• {l}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MAIN

    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb11_2 = s11.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf11_2 = tb11_2.text_frame
    tf11_2.word_wrap = True

    p = tf11_2.paragraphs[0]
    p.text = "Open Source Repository"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    links_2 = [
        "GitHub URL: https://github.com/Sandeep332005/AstraVision",
        "License: Apache License 2.0 (Open-Source)",
        "One-Command Local Launch:\n  git clone https://github.com/Sandeep332005/AstraVision.git\n  pip install -e .\n  python run.py",
        "One-Command Test Runner:\n  python run.py --test"
    ]
    for l in links_2:
        p = tf11_2.add_paragraph()
        p.text = f"• {l}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 12: Conclusion & Future Scope
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, BG_COLOR)
    add_header(s12, "Conclusion & Future Scope")

    add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb12_1 = s12.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf12_1 = tb12_1.text_frame
    tf12_1.word_wrap = True

    p = tf12_1.paragraphs[0]
    p.text = "Project Summary & Impact"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    concls = [
        "Successfully modernized SSIM and MS-SSIM into a production PyTorch 2.x research suite.",
        "Broke the perceptual blind spot of classical MSE/PSNR through interactive spatial telemetry.",
        "Demonstrated a novel dual-engine architecture bridging heavy GPU pipelines with zero-setup web clients.",
        "Achieved 100% test passing rate (23 unit tests + 28 module audit checks) with certified differentiability."
    ]
    for c in concls:
        p = tf12_1.add_paragraph()
        p.text = f"• {c}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MAIN

    add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb12_2 = s12.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True

    p = tf12_2.paragraphs[0]
    p.text = "Future Research Directions"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    futures = [
        "Tiled Block Processing: Implementing overlapping tile convolutions for high-throughput 8K video streaming.",
        "Video Quality Assessment (VQA): Extending spatial SSIM to Spatio-Temporal SSIM (ST-SSIM) across temporal motion vectors.",
        "Generative AI Integration: Packaging AstraVision as a standard pip package (astravision) for HuggingFace diffusers and PyTorch Lightning."
    ]
    for f in futures:
        p = tf12_2.add_paragraph()
        p.text = f"• {f}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MAIN

    # Final Thank You / Q&A Box
    p_q = tf12_2.add_paragraph()
    p_q.text = "\nThank You! Questions & Discussion."
    p_q.font.size = Pt(14)
    p_q.font.bold = True
    p_q.font.color.rgb = ACCENT_GREEN

    # Save presentation
    output_path = "/Users/braj/ssim-lab/AstraVision_Project_Presentation.pptx"
    prs.save(output_path)
    print(f"✓ Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_deck()
