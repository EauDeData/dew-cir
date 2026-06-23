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

### VIT ###
class ViTSpecialistModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(ViTSpecialistModel, self).__init__()

        self.base_model = torchvision.models.vit_b_32(pretrained=True)
        self.base_model.heads = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, output_dim, bias=True)
        )
    def forward(self, x):
        x_feats = self.base_model(x)

        return x_feats

### RESNET ###
class ResNetSpecialistModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(ResNetSpecialistModel, self).__init__()

        self.base_model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, output_dim, bias=True)
        )

    def forward(self, x):
        x_feats = self.base_model(x)

        return x_feats

class ResNetConditionedToYear(torch.nn.Module):
    def __init__(self, specialist_model: SpecialistModel, output_dim):
        pass
    def forward(self, images, years):
        pass


### VGG ###
class VGGSpecialistModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(VGGSpecialistModel, self).__init__()

        self.base_model = torchvision.models.vgg19_bn(
            weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        )

        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),

            torch.nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        x_feats = self.base_model(x)

        return x_feats

class ConditionedToYear(torch.nn.Module):
    def __init__(self, specialist_model: SpecialistModel, output_dim, origin_model = 'convnext'):
        super(ConditionedToYear, self).__init__()
        self.feature_extractor = specialist_model
        if origin_model == 'convnext':
            self.feature_extractor.base_model.classifier = torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Flatten(),
            ))
        elif origin_model == 'vit':
            self.feature_extractor.base_model.heads = torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(768, 1024, bias=True),
            ))
        elif origin_model == 'vgg':
            self.feature_extractor.base_model.classifier = torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(25088, 1024, bias=True),
            ))
        elif origin_model == 'resnet':
            self.feature_extractor.base_model.fc = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2048, 1024)))

        # This is like an embedding, but this way I dont have to change the dataloader
        self.year_encoder = torch.nn.Embedding(output_dim, 1024)


    def forward(self, images, years):
        x_images = self.feature_extractor(images)
        # x_images = F.normalize(x_images, p=2, dim=1)
        if not years is None:
            x_years = self.year_encoder(years)

            return x_images,  x_images + x_years
        return x_images, None
