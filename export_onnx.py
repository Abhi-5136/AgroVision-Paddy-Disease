import torch
from models.cross_attention_net import AgroVisionNet

def export_to_onnx():
    model = AgroVisionNet(num_classes=10)
    model.load_state_dict(torch.load("agrovision_best.pth", map_location="cpu"))
    model.eval()

    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_weather = torch.randn(1, 7, 7)

    torch.onnx.export(
        model,
        (dummy_img, dummy_weather),
        "agrovision.onnx",
        input_names=["image", "weather"],
        output_names=["logits", "attention_weights"],
        dynamic_axes={"image": {0: "batch_size"}, "weather": {0: "batch_size"}},
        opset_version=14
    )
    print("Exported ONNX model successfully to agrovision.onnx")

if __name__ == "__main__":
    export_to_onnx()