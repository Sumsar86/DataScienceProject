from sklearn.preprocessing import LabelEncoder


class LabelProcessor:
    def __init__(self):
        self.order = ["Poor", "Needs Improvement", "Satisfactory", "Good", "Excellent"]
        self.label_to_int = {label: i for i, label in enumerate(self.order)}
        self.int_to_label = {i: label for i, label in enumerate(self.order)}

    def encode_labels(self, y):
        return y.map(self.label_to_int)

    def decode_labels(self, y_int):
        return y_int.map(self.int_to_label)
'''
class LabelProcessor:
    def __init__(self):
        self.le = LabelEncoder()
    
    def encode_labels(self, y):
        y_encoded = self.le.fit_transform(y)
        return y_encoded

    def decode_labels(self, y):
        y_labels = self.le.inverse_transform(y)
        return y_labels
'''