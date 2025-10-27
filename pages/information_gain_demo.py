import math
from collections import Counter, defaultdict

data = [
    {"Outlook": "Sunny",    "Windy": False, "Play": "No"},
    {"Outlook": "Sunny",    "Windy": True,  "Play": "No"},
    {"Outlook": "Overcast", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain",     "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain",     "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain",     "Windy": True,  "Play": "No"},
    {"Outlook": "Overcast", "Windy": True,  "Play": "Yes"},
    {"Outlook": "Sunny",    "Windy": False, "Play": "No"},
    {"Outlook": "Sunny",    "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain",     "Windy": False, "Play": "Yes"},
    {"Outlook": "Sunny",    "Windy": True,  "Play": "Yes"},
    {"Outlook": "Overcast", "Windy": True,  "Play": "Yes"},
    {"Outlook": "Overcast", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain",     "Windy": True,  "Play": "No"},
]

def entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * math.log(p, 2)
    return H

def information_gain(dataset, attr, target="Play"):
    parent_labels = [row[target] for row in dataset]
    H_parent = entropy(parent_labels)

    splits = defaultdict(list)
    for row in dataset:
        splits[row[attr]].append(row[target])

    n = len(dataset)
    H_children = 0.0
    details = []
    for v, labels in splits.items():
        w = len(labels) / n
        Hv = entropy(labels)
        H_children += w * Hv
        details.append((v, len(labels), w, Hv))

    IG = H_parent - H_children
    return H_parent, details, H_children, IG

if __name__ == "__main__":
    attrs = ["Outlook", "Windy"]
    for a in attrs:
        H_parent, details, H_children, IG = information_gain(data, a, "Play")
        print(f"=== Thuộc tính: {a} ===")
        print(f"H(parent) = {H_parent:.4f}")
        for v, cnt, w, Hv in details:
            print(f"  - Giá trị {a} = {v}: count={cnt}, weight={w:.3f}, H(child)={Hv:.4f}")
        print(f"H(children) = {H_children:.4f}")
        print(f"Information Gain = {IG:.4f}\n")