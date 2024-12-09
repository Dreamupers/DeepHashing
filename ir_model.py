import torch
import torch.nn as nn
import torchvision.models as models

from tensorboardX import SummaryWriter

class BaseIRModel(nn.Module):
    def __init__(self, hash_bit, backbone_frozen=True, backbone="alexnet"):
        super().__init__()
        models_dims = {
            "alexnet": 4096,   #* acc@1: 56.522, acc@5: 79.066, GFLOPS: 0.71,  Flie size: 233.1 MB
            "vgg16": 4096,     #* acc@1: 71.592, acc@5: 90.382, GFLOPS: 15.47, Flie size: 527.8 MB
            "resnet50": 2048,  #* acc@1: 76.13,  acc@5: 92.862, GFLOPS: 4.09,  Flie size: 97.8  MB
            "vit_b_16": 768    #* acc@1: 81.072, acc@5: 95.318, GFLOPS: 17.56, Flie size: 330.3 MB
        }
        assert backbone in models_dims.keys()

        self.features = eval(f"models.{backbone}")(weights="IMAGENET1K_V1")
        self.featdim = models_dims[backbone]
        self.proj = nn.Linear(self.featdim, hash_bit)
        
        if backbone_frozen:
            for param in self.features.parameters():
                param.requires_grad = False

        self.alpha = 1.0
        self.global_step = 0
        self.get_feat = eval(f"self.get_feat_{backbone}")
    
    def get_feat_vit_b_16(self, imgs):
        x = self.features._process_input(imgs)
        n = x.shape[0]
        batch_class_token = self.features.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.features.encoder(x)
        x = x[:, 0]
        return x
    
    def get_feat_alexnet(self, imgs):
        x = self.features.features(imgs)
        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.features.classifier[:-1](x) #* [Dropout, Linear, ReLU, Dropout, Linear, ReLU, w/o Linear]
        return x

    def get_feat_vgg16(self, imgs):
        x = self.features.features(imgs)
        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.features.classifier[:-2](x) #* [Linear, ReLU, Dropout, Linear, ReLU, w/o Dropout, Linear]
        return x
    
    def get_feat_resnet50(self, imgs):
        x = self.features.conv1(imgs)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)
        x = self.features.layer1(x)
        x = self.features.layer2(x)
        x = self.features.layer3(x)
        x = self.features.layer4(x)
        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_code(self, imgs):
        feats = self.get_feat(imgs)
        h = self.proj(feats)
        b = torch.tanh(self.alpha*h)
        return feats, h, b

    def train_step(self):
        raise NotImplementedError

    def train_epoch_start(self, epoch):
        pass

    def train_epoch_end(self, epoch):
        pass