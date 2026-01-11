class DictDataWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            x, y, metadata = batch
            style = metadata[:, 0]

            yield {"image": x, "label": y, "style": style}
