from sklearn.preprocessing import LabelEncoder

class LabelProcessor:
    def __init__(self):
        self.le = LabelEncoder()
    
    def encode_labels(self, y):
        y_encoded = self.le.fit_transform(y)
        return y_encoded

    def decode_labels(self, y):
        y_labels = self.le.inverse_transform(y)
        return y_labels