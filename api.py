# Nombre: Dincia Ciprian Santana
# Matrícula: 24-EISN-2-027
# Proyecto:Clasificador de Basura Inteligente (Eco-Scanner)

import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import os
import json

# tiempo en descomponerse
datos_ambientales = {
    "battery": {"nombre": "Batería / Pila", "tiempo": "500 a 1,000 años"},
    "biological": {"nombre": "Desechos Biológicos", "tiempo": "Semanas a meses"},
    "cardboard": {"nombre": "Cartón", "tiempo": "1 año"},
    "glass": {"nombre": "Vidrio", "tiempo": "4,000 años o más"},
    "metal": {"nombre": "Metal", "tiempo": "10 a 100 años"},
    "paper": {"nombre": "Papel", "tiempo": "2 a 5 meses"},
    "plastic": {"nombre": "Plástico", "tiempo": "100 a 1,000 años"}
}
#clases
clases = ["battery", "biological", "cardboard", "glass", "metal", "paper", "plastic"]
num_clases = len(clases)


#  la carpeta contenedores de forma automática
base_path = os.path.join(os.path.dirname(__file__), "contenedores")

contenedores = {
    "plastic": os.path.join(base_path, "plastic.jpg"),
    "paper": os.path.join(base_path, "paper.jpg"),
    "cardboard": os.path.join(base_path, "paper.jpg"),#estara en el mismo contenedor que paper
    "glass": os.path.join(base_path, "glass.jpg"),
    "metal": os.path.join(base_path, "metal.jpg"),
    "biological": os.path.join(base_path, "biological.jpg"),
    "battery": os.path.join(base_path, "metal.jpg"),#estara en el mismo contenedor que metal 

}
# Dispositivo
device = torch.device("cpu")

# Cargar modelo
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_clases)

model.load_state_dict(torch.load("modelo.pth", map_location=device))
model.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función de predicción
def predict(img):
    if img is None:
        return "Sube una imagen", None

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    pred_idx = torch.argmax(probs).item()
    pred_class = clases[pred_idx]
    confianza = probs[pred_idx].item() * 100

  
    info = datos_ambientales.get(pred_class, {"nombre": pred_class, "tiempo": "No disponible"})

    texto = (f"MATERIAL: {info['nombre'].upper()}\n"
             f"CONFIANZA: {confianza:.2f}%\n"
             f"TIEMPO DE DEGRADACIÓN: {info['tiempo']}\n"
             f"\n"
             f"Depositar en su contenedor correspondiente.")

   
    contenedor_path = contenedores.get(pred_class, None)
    
    contenedor_img = Image.open(contenedor_path) if contenedor_path else None

    return texto, contenedor_img



# Definimos el estilo para que la imagen no sea gigante
css = ".contenedor-img { max-height: 650px !important; width: auto !important; margin: 0 auto !important; }"

with gr.Blocks(css=css, title="Eco-Scanner") as demo:
    gr.Markdown("#  Eco-Scanner")
    gr.Markdown("### Clasificador inteligente de residuos para reciclaje")
    
    with gr.Row():
        # Columna de la Izquierda: Entrada
        with gr.Column():
            input_img = gr.Image(type="pil", label="Sube o toma una foto")
            btn_run = gr.Button(" CLASIFICAR RESIDUO", variant="primary")
        
        # Columna de la Derecha: Resultados
        with gr.Column():
            output_text = gr.Text(label="Resultado")
            output_img = gr.Image(label="Contenedor recomendado", elem_classes=["contenedor-img"])

    # Conectamos el botón con la función predict
    btn_run.click(fn=predict, inputs=input_img, outputs=[output_text, output_img])

if __name__ == "__main__":
    demo.launch()