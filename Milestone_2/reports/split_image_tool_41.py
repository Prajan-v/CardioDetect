from PIL import Image
from pathlib import Path

base_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/reports')
input_path = base_dir / 'code_field_patterns.png'
output1 = base_dir / 'code_field_patterns_left.png'
output2 = base_dir / 'code_field_patterns_right.png'

if input_path.exists():
    img = Image.open(input_path)
    width, height = img.size
    
    # Split vertically (top half vs bottom half)
    mid_point = height // 2
    
    # Top half
    img1 = img.crop((0, 0, width, mid_point))
    img1.save(output1)
    
    # Bottom half
    img2 = img.crop((0, mid_point, width, height))
    img2.save(output2)
    
    print(f"Split image {input_path.name} ({width}x{height}) into:")
    print(f"  {output1.name} ({img1.size})")
    print(f"  {output2.name} ({img2.size})")
else:
    print(f"Input file not found: {input_path}")
