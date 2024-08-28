

class BaseStorage:
    def __init__(self, img_emb):
        self.img_emb = img_emb

    def storage(self):
        return self.img_emb
