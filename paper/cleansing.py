import pandas as pd

def capitalize_first_word_only(text):
    if not isinstance(text, str) or not text.strip():
        return text
    words = text.split()
    first = words[0].capitalize()
    rest = [word.lower() for word in words[1:]]
    return ' '.join([first] + rest)

input_file = 'data_review.csv'
output_file = 'output.csv'

df = pd.read_csv(input_file)

if 'review' in df.columns:
    df['review'] = df['review'].apply(capitalize_first_word_only)

df.to_csv(output_file, index=False)

print(f'File hasil sudah disimpan ke {output_file}')
