import torch.utils.data


def CreateDataset(opt):
    dataset = None
    from data.cd_dataset import ChangeDetectionDataset
    dataset = ChangeDetectionDataset()

    print("dataset [%s] was created" % (opt.dataset))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.phase != 'test',
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader