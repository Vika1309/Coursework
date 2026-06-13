import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Розпізнавач", page_icon="🔢")
st.title("Розпізнавання рукописних цифр (MNIST)")
st.write("Намалюйте цифру від 0 до 9 у чорному квадраті нижче.")

@st.cache_resource
def load_cnn_model():
    return load_model("mnist_cnn.h5")

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Помилка завантаження моделі: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Полотно для малювання")
    
    canvas_result = st_canvas(
        stroke_width=25,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("### Результат розпізнавання")
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

        if st.button("Розпізнати цифру"):

            if cv2.countNonZero(gray_image) == 0:
                st.warning("Спочатку намалюйте цифру!")
            else:
               
               
                coords = cv2.findNonZero(gray_image)
                x, y, w, h = cv2.boundingRect(coords)
                digit = gray_image[y:y+h, x:x+w]

                
                if w > h:
                    new_w = 20
                    new_h = int((20 / w) * h)
                else:
                    new_h = 20
                    new_w = int((20 / h) * w)
                
               
                resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

                
                pad_top = (28 - new_h) // 2
                pad_bottom = 28 - new_h - pad_top
                pad_left = (28 - new_w) // 2
                pad_right = 28 - new_w - pad_left
                
                padded_digit = cv2.copyMakeBorder(resized_digit, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

                
                normalized_image = padded_digit / 255.0
                reshaped_image = normalized_image.reshape(1, 28, 28, 1)
                

                
                predictions = model.predict(reshaped_image)
                predicted_digit = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                st.success(f"Це цифра: **{predicted_digit}**")
                st.info(f"Впевненість моделі: **{confidence:.2f}%**")
                
                st.bar_chart(predictions[0])