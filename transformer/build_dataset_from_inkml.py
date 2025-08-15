import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import csv
import argparse
import math
from tqdm import tqdm  # For progress bar

# InkML namespace
NS = '{http://www.w3.org/2003/InkML}'

def parse_inkml(file_path):
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def get_all_traces(root):
    import re
    traces = {}
    for trace in root.findall(f'.//{NS}trace'):
        trace_id = trace.get('id')
        txt = (trace.text or "").strip()
        points = []

        if txt:
            # Normalize newlines; split points on commas
            for token in txt.replace('\n', ' ').replace(';', ' ').split(','):
                token = token.strip()
                if not token:
                    continue

                # Fast path: split on whitespace, take first two
                parts = [s for s in token.split() if s]
                parsed = False
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        points.append((x, y))
                        parsed = True
                    except ValueError:
                        parsed = False

                # Fallback: regex float extraction (handles weird separators)
                if not parsed:
                    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', token)
                    if len(nums) >= 2:
                        try:
                            points.append((float(nums[0]), float(nums[1])))
                        except Exception:
                            pass

        traces[trace_id] = points
    return traces


def extract_symbols(root, traces, output_dir, base_filename, image_size=64, thickness=2):
    symbols = []
    symbol_id = 0

    def process_trace_group(group, parent_label=''):
        nonlocal symbol_id
        # Robust label extraction: handle missing/empty text
        label = parent_label or 'unknown'
        label_elem = group.find(f'{NS}annotation[@type="truth"]')
        if label_elem is not None:
            txt = label_elem.text
            if isinstance(txt, str):
                txt = txt.strip()
                if txt:
                    label = txt

        
        sub_groups = group.findall(f'{NS}traceGroup')
        if sub_groups:
            for sub in sub_groups:
                process_trace_group(sub, label)
        else:
            trace_refs = [tv.get('traceDataRef') for tv in group.findall(f'{NS}traceView')]
            if not trace_refs:
                print(f"Skipping symbol {symbol_id} due to no trace references")
                return
            
            all_points = []
            for ref in trace_refs:
                if ref in traces:
                    all_points.extend(traces[ref])
            
            if not all_points:
                print(f"Skipping symbol {symbol_id} due to no points")
                return
            
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
            
            # Calculate width and height
            width = max_x - min_x
            height = max_y - min_y
            
            # Handle degenerate cases by setting a minimum size
            min_size = 2.0
            if width <= 0:
                width = min_size
                max_x = min_x + min_size
            if height <= 0:
                height = min_size
                max_y = min_y + min_size
            
            # Scale to fit image_size with padding
            scale = min((image_size - 10) / width, (image_size - 10) / height)
            offset_x = (image_size - width * scale) / 2 - min_x * scale
            offset_y = (image_size - height * scale) / 2 - min_y * scale
            
            img = Image.new('L', (image_size, image_size), color=255)
            draw = ImageDraw.Draw(img)
            
            for ref in trace_refs:
                points = traces[ref]
                scaled_points = [(int(p[0] * scale + offset_x), int(p[1] * scale + offset_y)) for p in points]
                for i in range(len(scaled_points) - 1):
                    draw.line(scaled_points[i:i+2], fill=0, width=thickness)
            
            # Use base filename with symbol index
            filename = f"{base_filename}_{symbol_id}.png"
            img_path = os.path.join(output_dir, filename)
            try:
                img.save(img_path)
                if os.path.exists(img_path):
                    symbols.append((filename, label))
                else:
                    print(f"Failed to save PNG: {img_path}")
            except Exception as e:
                print(f"Error saving PNG {img_path}: {e}")
            
            symbol_id += 1

    for tg in root.findall(f'.//{NS}traceGroup'):
        process_trace_group(tg)
    
    return symbols

def process_directory(input_dir, output_dir, label_file, image_size, thickness):
    # Define input and output folder pairs
    folders = [('testINKML', 'testPNG'), ('trainINKML', 'trainPNG'), ('validINKML', 'validPNG')]
    
    for input_folder, output_folder in folders:
        input_path = os.path.join(input_dir, input_folder)
        output_path = os.path.join(output_dir, output_folder)
        os.makedirs(output_path, exist_ok=True)
        
        # Get list of InkML files in the current folder
        inkml_files = [f for f in os.listdir(input_path) if f.endswith('.inkml')]
        
        # Add progress bar
        folder_symbols = []
        for filename in tqdm(inkml_files, desc=f"Processing {input_folder}"):
            file_path = os.path.join(input_path, filename)
            root = parse_inkml(file_path)
            if root is None:
                continue
            traces = get_all_traces(root)
            # Use the InkML filename without extension as the base
            base_filename = os.path.splitext(filename)[0]
            symbols = extract_symbols(root, traces, output_path, base_filename, image_size, thickness)
            # Store only the base filename, not the full path with output_folder
            folder_symbols.extend([(s[0], s[1]) for s in symbols])
        
        print(f"Processed {len(folder_symbols)} symbols for {input_folder}")
        
        # Write labels to a separate CSV file for each folder
        label_file_path = os.path.join(output_dir, f"{input_folder.replace('INKML', 'labels')}.csv")
        with open(label_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])
            writer.writerows(folder_symbols)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert InkML files to PNG images with symbol-level labels.')
    parser.add_argument('input_dir', help='Directory containing InkML folders (e.g., CROHME_labeled_2016)')
    parser.add_argument('output_dir', help='Directory to save PNG images and labels')
    parser.add_argument('label_file', help='Base path for CSV label files (ignored, but required for compatibility)')
    parser.add_argument('--image_size', type=int, default=64, help='Size of output square images')
    parser.add_argument('--thickness', type=int, default=2, help='Line thickness for rendering')
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.label_file, args.image_size, args.thickness)