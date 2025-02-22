import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence
import gradio as gr
import yaml
import torch


def load_model(config, device):
    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(device)

    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value

        text2cad.load_state_dict(pretrained_dict, strict=False)
    text2cad.eval()
    return text2cad

def test_model(model, text, config, device):
    
    if not isinstance(text, list):
        text = [text]

    pred_cad_seq_dict = model.test_decode(
        texts=text,
        maxlen=MAX_CAD_SEQUENCE_LENGTH,
        nucleus_prob=0,
        topk_index=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # pred_cad_seq_dict: dict_keys(['cad_vec', 'flag_vec', 'index_vec'])  分别代表了什么
    # print("pred_cad_seq_dict[\"cad_vec\"]", pred_cad_seq_dict["cad_vec"])
    print('pred_cad_seq_dict:',pred_cad_seq_dict.keys())
    
    # 向量 -> command sequence -> 渲染3dmodel
    try:
        # 这里是怎么从vector解码到command sequence的
        pred_cad = CADSequence.from_vec(
            pred_cad_seq_dict["cad_vec"][0].cpu().numpy(),
            bit=N_BIT,
            post_processing=True,
        ).create_mesh()
        print("pred_cad: ", pred_cad, type(pred_cad))
        print("pred_cad.mesh: ", pred_cad.mesh, type(pred_cad.mesh))
        return pred_cad.mesh, pred_cad
    except Exception as e:
        return None
    
    '''
    - Sketch:
       - CoordinateSystem:
            - Rotation Matrix [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            - Translation [0. 0. 0.]
       - Face:
          - Loop: Start Point: [0.0029, 0.0029], Direction: counterclockwise
              - Circle: center([0.3765 0.3765]),             radius(0.3735), pt1 [0.37647059 0.75      ]

          - Loop: Start Point: [0.1794, 0.1794], Direction: counterclockwise
              - Circle: center([0.3765 0.3765]),             radius(0.1971), pt1 [0.37647059 0.57352941]


    - ExtrudeSequence: (extent_one: 0.453125, extent_two: 0.0, boolean: 0, sketch_size: 0.75) Euler Angles [0. 0. 0.]
    '''

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data



config_path = "../Cad_VLM/config/inference_user_input.yaml"
config = parse_config_file(config_path)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = load_model(config, device)
OUTPUT_DIR="output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def genrate_cad_model_from_text(text):
    global model, config
    mesh, *extra = test_model(model=model, text=text, config=config, device=device)
    if mesh is not None:
        output_path = os.path.join(OUTPUT_DIR, "output.stl")
        mesh.export(output_path) # 生成3d模型保存文件
        return output_path
    else:
        raise Exception("Error generating CAD model from text")


examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "The CAD model features a cylindrical object with a cylindrical hole in the center.",
    "The CAD model features a rectangular metal plate with four holes along its length."
]

title = "Text2CAD demo测试"
description = """

<div style="display: flex; justify-content: center; gap: 10px; align-items: center;">

</div>
"""

# Create the Gradio interface
demo = gr.Interface(
    fn=genrate_cad_model_from_text,
    inputs=gr.Textbox(label="指令", placeholder="Enter a text prompt here"),
    outputs=gr.Model3D(clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"), # 3d渲染插件
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(), 
)

if __name__ == "__main__":
    demo.launch(share=True)
