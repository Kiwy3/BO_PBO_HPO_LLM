import datasets

ds_name = "databricks/databricks-dolly-15k"
#ds = datasets.load_dataset()

print(datasets.get_dataset_split_names(ds_name))


"""
class LLMDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.Alpaca2k(self.data_dir, train=True, download=True)
        datasets.Alpaca2k(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.Alpaca2k(
            root=self.data_dir, 
            train=True,
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        """