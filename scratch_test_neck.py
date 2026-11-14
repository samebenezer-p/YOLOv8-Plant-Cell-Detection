import torch
import torch.nn as nn
from ultralytics import YOLO

class Select(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x[self.index]

class DummyFusionNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.conv4 = nn.Conv2d(512, 256, 1)
        self.conv5 = nn.Conv2d(1024, 256, 1)
        
    def forward(self, features):
        p3, p4, p5 = features
        return self.conv3(p3), self.conv4(p4), self.conv5(p5)

def inject(model):
    # original backbone: 0 to 9
    # original neck: 10 to 21
    # original detect: 22
    seq = model.model.model
    
    # create new layers
    fusion = DummyFusionNeck()
    fusion.f = [4, 6, 9]
    fusion.i = 10
    
    sel0 = Select(0)
    sel0.f = 10
    sel0.i = 11
    
    sel1 = Select(1)
    sel1.f = 10
    sel1.i = 12
    
    sel2 = Select(2)
    sel2.f = 10
    sel2.i = 13
    
    det = seq[22]
    det.f = [11, 12, 13]
    det.i = 14
    
    # replace
    new_seq = nn.ModuleList([
        *seq[:10],
        fusion,
        sel0, sel1, sel2,
        det
    ])
    model.model.model = new_seq
    model.model.save = [4, 6, 9, 10, 11, 12, 13]

model = YOLO('yolov8n.pt')
inject(model)
model.info()

x = torch.randn(1, 3, 640, 640)
out = model.model(x)
print("Forward pass successful, out len:", len(out))
