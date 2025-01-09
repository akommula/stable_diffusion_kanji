import xml.etree.ElementTree as ET
from cairosvg import svg2png
from PIL import Image

from tqdm import tqdm
import shutil
import json
import os

def process_svg_to_png(input_svg_path):
    tree = ET.parse(input_svg_path)
    root = tree.getroot()
        
    for g_elem in root.findall('.//{http://www.w3.org/2000/svg}g'):        
        elem_id = g_elem.get('id', '')
        if 'StrokeNumbers' in elem_id:
            parent = root.find('.//*[@id="{}"]/../..'.format(elem_id))
            if parent is not None:
                parent.remove(g_elem)
            else:
                root.remove(g_elem)
    
    paths = root.findall(".//*[@style]")
    for path in paths:
        style = path.get('style')
        if style:
            new_style = style.replace("stroke:#000000", "stroke:#000000").replace("fill:#808080", "fill:#000000")
            path.set('style', new_style)
    
    modified_svg = ET.tostring(root, encoding='utf-8')
    
    png_data = svg2png(bytestring=modified_svg, output_width=128, output_height=128)
    return png_data


def extract_kanji_meanings(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        kanji_meanings = []
        for character in root.findall('.//character'):  # Using xpath to find all character elements
            kanji = character.find('literal').text
            
            meanings = []
            rmgroup = character.find('.//rmgroup')
            if rmgroup is not None:
                for meaning in rmgroup.findall('meaning'):
                    if meaning.text and 'm_lang' not in meaning.attrib:
                        meanings.append(meaning.text)
            
            if len(meanings) > 0:
                kanji_meanings.append((kanji, meanings))

        return kanji_meanings
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

def build_kanji_index(output_dir = 'kanji_dataset'):
    index_path = 'kanjivg/kvg-index.json'
    kanji_xml_path = 'kanjidic2.xml'    
    kanji_svg_dir = 'kanjivg/kanji'
    
    # Extract definitions per kanji character
    get_def = {}
    
    kanji_and_definitions = extract_kanji_meanings(kanji_xml_path)
    
    print(f"Number of kanji loaded: {len(kanji_and_definitions)}")
    for char, definitions in kanji_and_definitions:
        print(f"Kanji: {char}; Definitions: {definitions}")
        get_def[char] = definitions
    
    with open(index_path, 'r') as file:
        data = json.load(file)
        
        for kanji_char in tqdm(data, desc = "Processing Kanji Characters..."):
            if kanji_char not in get_def:
                continue
            
            files = data[kanji_char]
            svg_path = None 
            
            for file_name in files:
                if '-' not in file_name:
                    svg_path = file_name                    
                    break
            
            if svg_path is None:
                continue
            
            kanji_id = svg_path.split('.svg')[0]
            kanji_png = process_svg_to_png(kanji_svg_dir + '/' + svg_path)  
            kanji_definitions = get_def[kanji_char]
            
            dir = output_dir + f"/{kanji_id}"
            
            if os.path.exists(dir):
                shutil.rmtree(dir)    
            os.makedirs(dir)
            
            with open(dir + f"/{kanji_id}.png", 'wb') as f:
                f.write(kanji_png)
            with open(dir + f"/{kanji_id}_definitions.json", 'w') as f:
                json.dump(kanji_definitions, f, indent = 2)
            
            
if __name__ == '__main__':
    build_kanji_index()
    