import torch
import torch.nn as nn
from ultralytics import YOLO

class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 1, 1)
        self.latest_heatmap = None
        
    def forward(self, x):
        self.latest_heatmap = self.conv(x)
        # Return x unmodified so the rest of YOLO works normally
        return x

def inject(model):
    seq = model.model.model
    # In yolov8n, P3 is output by layer 15 (C2f)
    # Detect is layer 22, reading from [15, 18, 21]
    
    boundary = BoundaryHead()
    boundary.i = 16 # we will insert it after 15
    boundary.f = 15 # reads from 15
    
    # We need to insert it and update indices
    # However, if we just set it up to intercept layer 15's output:
    # Actually, layer 15 is a C2f. If we just wrap layer 15!
    class InterceptWrapper(nn.Module):
        def __init__(self, original_module, boundary_head):
            super().__init__()
            self.original = original_module
            self.boundary = boundary_head
            self.i = original_module.i
            self.f = original_module.f
            self.type = original_module.type if hasattr(original_module, 'type') else 'Intercept'
            
        def forward(self, x):
            out = self.original(x)
            self.boundary(out) # saves latest_heatmap
            return out
            
    seq[15] = InterceptWrapper(seq[15], boundary)
    return boundary

model = YOLO('yolov8n.pt')
head = inject(model)
model.model.train()
x = torch.randn(2, 3, 640, 640)
out = model.model(x)
print("Forward pass successful!")
print("Boundary heatmap shape:", head.latest_heatmap.shape)
