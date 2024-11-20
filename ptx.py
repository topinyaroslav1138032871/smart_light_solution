# import torch

# s_model = torch.load('yolo11s.pt')
# t_model = torch.load('best.pt')

# class_ind = 0

# s_weights = s_model['model'][-1].bias(class_ind)
# s_biases = s_model['model'][-1].weight(class_ind)

# t_h = t_model['model'][-1]
# t_h.bias = torch.cat([t_h.bias,s_biases.unsqueeze(0)],dim=0)
# t_h.weight = torch.cat([t_h.weight,s_weights.unsqueeze(0)],dim=0)

# t_model['nc']+=1

# torch.save(t_model,'updated.pt')

import torch
from ultralytics import YOLO

model = torch.load("templates/models/best1.pt", map_location="cpu")
for x in range(1,20):
    model["model"].names[x] = f"{x}"

torch.save(model, "save_best.pt")

model = YOLO(r"save_best.pt")
print(model.names)