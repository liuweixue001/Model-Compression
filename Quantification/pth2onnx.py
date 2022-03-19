import numpy as np
import torch
import onnxruntime
from torchvision.models import shufflenet_v2_x0_5
device = torch.device("cuda")

# ---------------- --------------------------------tensor2numpy-----------------------------------------------
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    # ------------------------------------------------定义模型--------------------------------------------------
    x = torch.randn(1, 3, 224, 224, device='cuda')
    model = shufflenet_v2_x0_5().to(device)
    model.load_state_dict(torch.load("./model/shufflenet.pth"))
    # ------------------------------------------------导出模型--------------------------------------------------
    torch.onnx.export(model,  # model
                      x,  # input
                      "./model/shufflenet.onnx",  # saved_model,
                      export_params=True,
                      opset_version=11,
                      input_names=["input"],
                      output_names=["output"])
                      # dynamic_axes={'input': {'0': 'batch_size'},
                      #               'output': {'0': 'batch_size'}})
    # ------------------------------------------------精度对比--------------------------------------------------
    model.eval()
    out = model(x)
    ort_session = onnxruntime.InferenceSession("./model/shufflenet.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-3, atol=1e-5)
    print(out)
    print(ort_outs)
    print("Excepted model has been test with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
