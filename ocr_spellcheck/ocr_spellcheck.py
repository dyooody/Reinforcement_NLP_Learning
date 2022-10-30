# %%
import cv2
from PIL import Image

import nltk
import re
from textblob import Word

# %%
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# %%
#test_image = cv2.imread("test-image-for-recognition.png")
test_image = cv2.imread("incorrect2.png")
#test_image = cv2.imread("flight.JPG")

# %%
extracted_string = pytesseract.image_to_string(test_image)

# %%
print(extracted_string)

# %%
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
  # Removing html tags
  sentence = remove_tags(sen)
  # Remove punctuations and numbers
  sentence = re.sub('[^a-zA-Z]', ' ', sentence)
  # Single character removal
  sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
  # Removing multiple spaces
  sentence = re.sub(r'\s+', ' ', sentence)
  return sentence

# %%
def spell_check(word):
  word = Word(word)
  result = word.spellcheck()

  if word == result[0][0]:
    return word, True
  else:
    return result[0][0], False

# %%
def spell_check_for_sentence(extracted_string):
  extracted_string = preprocess_text(extracted_string)
  sentence = extracted_string.split(" ")
  sentence = [word.lower() for word in sentence]

  checked = []
  mis_spelled = []
  for word in sentence:
    if word != '':
      w, val = spell_check(word)
      checked.append(w)
      if val == False:
        mis_spelled.append(word)


  checked_sentence = ' '.join(checked)
  return checked_sentence, mis_spelled

# %%
final_sen, mis_spelled = spell_check_for_sentence(extracted_string)

# %%
print(final_sen)

print(mis_spelled)


