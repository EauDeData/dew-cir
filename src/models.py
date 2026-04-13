import torch
import torchvision

class SpecialistModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(SpecialistModel, self).__init__()
        print(output_dim)
        self.base_model = torchvision.models.convnext_base(weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, output_dim, bias=True)
        )
    def forward(self, x):
        x_feats = self.base_model(x)

        return x_feats


class ConditionedToYear(torch.nn.Module):
    def __init__(self, specialist_model: SpecialistModel, output_dim):
        super(ConditionedToYear, self).__init__()
        self.feature_extractor = specialist_model
        self.feature_extractor.base_model.classifier = torch.nn.Sequential(torch.nn.Sequential(
            torch.nn.Flatten(),
        ))

        # This is like an embedding, but this way I dont have to change the dataloader
        self.year_encoder = torch.nn.Embedding(output_dim, 1024)


    def forward(self, images, years):
        x_images = self.feature_extractor(images)
        # x_images = F.normalize(x_images, p=2, dim=1)
        if not years is None:
            x_years = self.year_encoder(years)

            return x_images,  x_images + x_years
        return x_images, None



