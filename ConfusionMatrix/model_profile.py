from torchstat import stat
from torchinfo import summary
from thop import profile, clever_format

import torch

# from torchvision.models import resnet152 as create_model
from torchvision.models import resnet50 as create_model


# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create model
model = create_model()
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 1000)

batch_size = 1

# Calculate the complexity of models

# --> torchinfo <--
# summary(model=model, input_size=(batch_size, 3, 224, 224))

# --> torchstat <--
# stat(model, (3, 224, 224))

# --> thop <--
my_input = torch.zeros((batch_size, 3, 224, 224)).to(device)
flops, params = profile(model.to(device), inputs=(my_input,))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)
