# User Input Validation Project

This program checks if the user-provided input data is valid based on the ID text and the user's declared country. It allows validation only for specific countries: Poland, Belgium, Czech Republic, China, and Egypt.

## Features
- Extracts text from an image using OCR (Tesseract)
- Validates the country name from the extracted text
- Provides feedback on whether the ID data is valid for the declared country

## Requirements
- OpenCV
- Tesseract-OCR
- pytesseract
- numpy

## Installation
Install required packages with:
pip install -r requirements.txt

# Usage
Run the program by specifying the declared country:
python main.py

# Example
Input: Image containing text with the country 'Poland'.

Output:
- Your country: Poland
- Country from text: Poland
- ID OK