'''
This program is checking if user input data is valid with ID. 
Only 2 countries are allowed to be checked. 
'''

import cv2
import numpy as np
import pytesseract

def check_country_from_text(text):
    """
    Checks if text contains name of a country from this list.

    Args:
        text (str): text to search for
    
    Returns:
        str: name of a country or None.
    """
    keywords = {
        'Poland': ['Poland', 'Polska', 'PL', 'Polish', 'Polskie'],
        'Belgium': ['Belgium', 'Belgique', 'België', 'BEL'],
    }

    for country, keylist in keywords.items():
        for word in keylist:
            if word.lower() in text.lower():
                return country

    return None

def countries(user_declared_country):
    country_languages = {
        'Belgium': 'eng+bel',
        'Poland': 'eng+pol',
    }

    return country_languages.get(user_declared_country, 'eng')

def rect(contours):
    largest_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        if area > max_area:
            largest_contour = approx
            max_area = area
    
    return max_area, largest_contour

def safe_ocr_results_to_file(file_path, results):
    """
    Saves results of OCR to a file.

    Args:
        file_path (str): path to a file
        results (list): results of words scan 
    """
    try: 
        with open(file_path, 'w', encoding='utf-8') as file:
            for result in results:
                file.write(f"Text: {result['text']}, Confidence: {result['conf']}\n")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

def process_image_and_ocr(user_declared_country):
    lang = countries(user_declared_country=user_declared_country)

    img = cv2.imread('images/bel.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    _, thresh1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    _, largest_contour = rect(contours)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], -1, (0, 255, 0), 2)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    (y, x, _) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy: bottomy + 1, topx: bottomx + 1, :]

    results = []
    config = "--psm 3 --oem 3"
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    img_data = pytesseract.image_to_data(
        out_rgb,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DATAFRAME,
    )

    for _, row in img_data.iterrows():
        text = str(row['text']).strip()
        conf = int(row['conf']) if 'conf' in img_data.columns else -1
        results.append({'text' : text, 'conf' : conf})

    safe_ocr_results_to_file('ocr_results.txt', results)

    img_conf_text = img_data[["conf", "text"]]
    img_valid = img_conf_text[img_conf_text["text"].notnull()]
    img_words = img_valid[img_valid["text"].str.len() > 1]

    confidence_level = 75
    img_conf = img_words[img_words["conf"] > confidence_level]
    predictions = img_conf["text"].to_list()

    text_from_id = " ".join(predictions)
    country_detected = check_country_from_text(text_from_id)

    return country_detected

def main():
    user_declared_country = input("Your country: ")
    country_detected = process_image_and_ocr(user_declared_country)

    if country_detected and country_detected.lower() == user_declared_country.lower():
        print(f"Country from text: {country_detected}")
        print("ID OK")
    else:
        print("Wrong ID")

main()