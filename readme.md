# Handwriting Synthesis ✍️


# Handwriting Synthesis ✍️

**⚠️ This project requires Python 3.6.13**  

This project is a implementation that reuses the [Handwriting Synthesis repository by Sean Vasquez](https://github.com/sjvasquez/handwriting-synthesis).  
It leverages a pretrained **Recurrent Neural Network (RNN)** to generate **realistic handwriting sequences** and exports them as **PNG files**.  

---

## 📦 Installation


```bash
git clone https://github.com/RedouaneH/Handwriting_generation.git
cd handwriting-synthesis
pip install -r requirements.txt
```

## 🚀 Quick Usage

```python

from IPython.display import Image as ipython_image, display
from generate_handwriting import generate_handwriting

output_path = 'img/hello_world.png'
text = "Hello World!"
bias = 1
color = "red"
style = 9
stroke_width = 2

generate_handwriting(
    text=text,
    stroke_width=stroke_width,
    bias=bias,
    color=color,
    style=style,
    output_path=output_path
)

display(ipython_image(filename=output_path))

```

![Hello World Handwriting](img/hello_world.png)


## 🎛️ Parameters Explanation

- **`bias`**: Controls the neatness of the generated handwriting. It is the **opposite of variance** — higher values produce neater, smoother handwriting, while lower values make it more irregular. Acceptable values are **≥ 0**.  

- **`style`**: Selects the handwriting style from the pretrained model. Each integer corresponds to a different style. Acceptable values are **between 0 and 210**.  
