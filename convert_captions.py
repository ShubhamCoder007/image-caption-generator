import json
from collections import defaultdict

def build_json(input_txt, output_json):
    captions = defaultdict(list)
    with open(input_txt, 'r') as f:
        for line in f:
            img_tok, caption = line.strip().split('\t')
            img_id, _idx   = img_tok.split('#')
            captions[img_id].append(caption)
    with open(output_json, 'w') as out:
        json.dump(captions, out)

if __name__ == "__main__":
    build_json(
      "Flickr8k.token.txt",
      "flickr8k_captions.json"
    )
    print("Wrote captions→flickr8k_captions.json")
