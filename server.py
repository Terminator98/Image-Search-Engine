import gradio as gr
import test

demo = gr.Interface(test.test ,
 gr.Image(type = "numpy") ,
 [gr.Image(shape=(400,400)) for i in range(5)])
demo.launch()