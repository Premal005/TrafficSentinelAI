import os
import re
from pathlib import Path
from fpdf import FPDF

class PresentationPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_margins(12, 20, 12)
        self.set_auto_page_break(False)

    def header(self):
        # Draw header banner
        self.set_fill_color(15, 23, 42) # Deep Slate (Navy Dark)
        self.rect(0, 0, 297, 16, "F")
        
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.text(12, 10, "FLIPKART × BTP | GRIDLOCK HACKATHON 2.0")
        
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(245, 158, 11) # Gold Accent
        self.text(205, 10, "PROTOTYPE PITCH: TRAFFICSENTINEL AI")
        
        # Gold underline
        self.set_draw_color(245, 158, 11)
        self.set_line_width(0.6)
        self.line(0, 16, 297, 16)

    def footer(self):
        # Draw footer line
        self.set_draw_color(226, 232, 240) # Slate light
        self.set_line_width(0.3)
        self.line(12, 198, 285, 198)
        
        self.set_text_color(148, 163, 184) # Slate gray
        self.set_font("Helvetica", "I", 8)
        self.text(12, 203, "TrafficSentinel AI - Winner's Codebase Submission")
        self.text(270, 203, f"Slide {self.page_no()}")

def clean_markdown_text(text: str) -> str:
    """Remove markdown syntax for clean PDF rendering."""
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("`", "")
    text = text.replace("➔", "->")
    # Replace unicode characters that standard PDF Helvetica font can't render
    text = text.replace("🗣️", "Speaker Notes:")
    text = text.replace("🎬", "")
    text = text.replace("🚦", "")
    text = text.replace("📌", "")
    text = text.replace("🗺️", "")
    text = text.replace("🛠️", "")
    text = text.replace("📂", "")
    text = text.replace("⚡", "")
    text = text.replace("📊", "")
    text = text.replace("🚀", "")
    text = text.replace("🙋", "")
    text = text.replace("➔", "->")
    text = text.replace("•", "-")
    text = text.replace("γ", "gamma")
    text = text.replace("θ", "theta")
    text = text.replace("iff", "<=>")
    text = text.replace("≤", "<=")
    text = text.replace("≥", ">=")
    text = text.replace("—", "-")
    
    # Replace math notations with plain text equivalent
    text = text.replace("$$\\text{RIDES} \\iff \\text{IoU}(B_p, B_v) \\ge 0.15 \\text{ and } \\text{center}_y(B_p) < \\text{center}_y(B_v)$$", 
                        "RIDES Edge if: IoU(Box_Person, Box_Vehicle) >= 0.15 AND Person_Center_Y < Vehicle_Center_Y")
    
    # Replace other potential unsupported characters
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text.strip()

def parse_markdown_slides(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split slides
    slides_raw = re.split(r'### Slide\s+', content)
    slides_data = []

    for raw in slides_raw[1:]:
        lines = raw.strip().split("\n")
        header = lines[0].strip()
        body = "\n".join(lines[1:])
        
        # Extract visual layout
        visual_match = re.search(r'\*\s+\*\*Visual Layout\*\*:(.*?)(?=\*\s+\*\*Key Takeaways\*\*:|\*\s+\*\*Presenter Script\*\*:|>|$)', body, re.DOTALL)
        visual_text = visual_match.group(1).strip() if visual_match else ""
        
        # Extract key takeaways
        takeaways_match = re.search(r'\*\s+\*\*Key Takeaways\*\*:(.*?)(?=\*\s+\*\*Presenter Script\*\*:|>|$)', body, re.DOTALL)
        takeaways_text = takeaways_match.group(1).strip() if takeaways_match else ""
        
        # Extract presenter script
        script_match = re.search(r'Presenter Script\*\*:\s*>\s*\*?"?(.*?)"?\$', body, re.DOTALL)
        if not script_match:
            # Fallback pattern
            script_match = re.search(r'>\s*\*?"?(.*?)"?$', body, re.DOTALL)
        
        script_text = script_match.group(1).strip() if script_match else ""
        
        # If script text has visual layout items or takeaways, clean them
        script_text = script_text.replace("\n> ", "\n")
        
        slides_data.append({
            "header": header,
            "visuals": visual_text,
            "takeaways": takeaways_text,
            "script": script_text
        })
        
    return slides_data

def generate_pdf(slides, output_path: Path):
    pdf = PresentationPDF()
    
    for idx, slide in enumerate(slides):
        pdf.add_page()
        
        # Draw Left Pane Title
        pdf.set_y(22)
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(15, 23, 42) # Dark Slate Blue
        slide_title = clean_markdown_text(slide["header"])
        pdf.cell(0, 8, f"SLIDE {idx+1}: {slide_title}", ln=True)
        pdf.ln(2)
        
        # Draw Two Columns
        # Column 1: Visuals & Takeaways (Width: 155mm)
        # Column 2: Presenter Script Card (Width: 105mm, X: 175mm)
        
        # --- LEFT COLUMN (Visuals & Takeaways) ---
        col1_w = 152
        
        # Visual Layout Box
        pdf.set_fill_color(248, 250, 252) # Light blue slate background
        pdf.set_draw_color(203, 213, 225) # Border Slate
        pdf.set_line_width(0.2)
        pdf.rect(12, 34, col1_w, 64, "FD")
        
        pdf.set_y(36)
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(40, 116, 240) # Flipkart Blue
        pdf.cell(0, 5, "VISUAL SLIDE LAYOUT DESIGN", ln=True)
        pdf.ln(1)
        
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(51, 65, 85) # Slate text
        
        visual_lines = clean_markdown_text(slide["visuals"]).split("\n")
        pdf.set_x(15)
        for line in visual_lines:
            line_str = line.strip().lstrip("- ").strip()
            if line_str:
                pdf.set_x(15)
                # Print bullet point
                pdf.write(4.5, f"- {line_str}\n")
        
        # Key Takeaways Box
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(12, 104, col1_w, 86, "FD")
        
        pdf.set_y(106)
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 41, 59) # Slate Dark
        pdf.cell(0, 5, "KEY TECHNICAL & BUSINESS TAKEAWAYS", ln=True)
        pdf.ln(1)
        
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(51, 65, 85)
        
        takeaway_lines = clean_markdown_text(slide["takeaways"]).split("\n")
        for line in takeaway_lines:
            line_str = line.strip().lstrip("- ").strip()
            if line_str:
                pdf.set_x(15)
                pdf.write(4.5, f"- {line_str}\n")
                
        # --- RIGHT COLUMN (Presenter Script Card) ---
        col2_x = 172
        col2_w = 113
        
        # Draw Presenter Card Background
        pdf.set_fill_color(241, 245, 249) # Light Gray Slate
        pdf.set_draw_color(148, 163, 184) # Darker Border
        pdf.set_line_width(0.3)
        pdf.rect(col2_x, 34, col2_w, 156, "FD")
        
        # Top banner for Presenter Notes
        pdf.set_fill_color(15, 23, 42) # Slate Dark
        pdf.rect(col2_x, 34, col2_w, 8, "F")
        
        pdf.set_y(35)
        pdf.set_x(col2_x + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 5, "SPEAKER SCRIPT & PRESENTATION NOTES")
        
        # Speaker notes text
        pdf.set_y(46)
        pdf.set_x(col2_x + 5)
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.set_text_color(30, 41, 59)
        
        script_cleaned = clean_markdown_text(slide["script"])
        # Format script text to stay inside boundaries (MultiCell)
        pdf.set_x(col2_x + 5)
        pdf.multi_cell(col2_w - 10, 5.5, f'"{script_cleaned}"')
        
    pdf.output(str(output_path))

if __name__ == "__main__":
    project_root = Path("E:/flipkart2r")
    deck_path = project_root / "docs/presentation_deck.md"
    pdf_output = project_root / "docs/presentation_deck.pdf"
    
    print("Parsing presentation deck markdown...")
    slides_data = parse_markdown_slides(deck_path)
    
    print("Generating beautifully formatted PDF deck...")
    generate_pdf(slides_data, pdf_output)
    print(f"Success! Presentation PDF saved at: {pdf_output}")
