from tkinter import Tk, Canvas, Button
from PIL import Image, ImageOps
from torchvision import transforms
import torch
from model import LeNet5  

class DigitRecognizer:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Digit Recognizer")
        self.model = self.load_model(model_path)
        self.root.geometry("500x500")
        self.canvas = Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        self.predict_button = Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

    def load_model(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LeNet5().to(device)
        model.device = device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def draw(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def preprocess_image(self, image):
        image = image.resize((32, 32)) 
        image = ImageOps.invert(image)  
        image = image.convert('L')  
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        image = transform(image).unsqueeze(0)
        return image


    def predict(self):
        # Guarda el canvas como una imagen
        self.canvas.postscript(file="tmp.eps", colormode="mono")
        img = Image.open("tmp.eps")
        # Muestra la imagen con PIL
        # img.show()
        img = self.preprocess_image(img)
        # Mueve la imagen al mismo dispositivo que el modelo
        img = img.to(self.model.device)
        # Realiza la predicción
        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
        # Muestra la predicción
        self.root.title("Predicted digit: " + str(predicted.item()))

        
        
if __name__ == "__main__":
    model_path = "path del modelo entrenado"
    root = Tk()  
    app = DigitRecognizer(root, model_path)  
    root.mainloop()