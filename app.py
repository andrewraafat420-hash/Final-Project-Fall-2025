from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- 1. إعدادات الموديلات (الترتيب الأبجدي الصحيح) ---
MODELS_CONFIG = {
    'cte': {
        'path': 'CTE_analyzer_model 2.h5', 
        'classes': {0: 'Stroke (جلطة)', 1: 'Normal (سليم)'},
        'target_size': (224, 224),
        'needs_rescaling': False, 
        'model': None
    },
    'malaria': {
        'path': 'Malaria_analyzer_model.h5', 
        'classes': {0: 'Parasitized (مصاب)', 1: 'Uninfected (سليم)'},
        'target_size': (180, 180),
        'needs_rescaling': False, # عطلناها لحل مشكلة الثبات الرقمي
        'model': None
    }
}

print("⏳ Loading Models into memory... Please wait.")
for key, config in MODELS_CONFIG.items():
    try:
        config['model'] = tf.keras.models.load_model(config['path'])
        print(f"✅ {key.upper()} Model Loaded!")
    except Exception as e:
        print(f"❌ Error loading {key} model: {e}")

# --- 2. دالة معالجة الصور ---
def preprocess_image(img_bytes, config):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(config['target_size'])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if config['needs_rescaling']:
        img_array = img_array / 255.0
    
    return img_array

# --- 3. الـ Route الرئيسي للتوقع ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    disease_type = request.form.get('disease_type')
    
    if file.filename == '' or disease_type not in MODELS_CONFIG:
        return jsonify({'error': 'Invalid request'})
    
    selected_config = MODELS_CONFIG[disease_type]
    
    if selected_config['model'] is None:
        return jsonify({'error': 'Model not ready'})

    try:
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes, selected_config)
        
        # تنفيذ التوقع (Inference)
        raw_prediction = selected_config['model'].predict(processed_img)
        
        # 🧪 "السر الهندي" لحل مشكلة الـ 641% والـ 139%:
        # دالة Softmax بتجبر مجموع الاحتمالات يكون 1 (يعني 100%)
        probabilities = tf.nn.softmax(raw_prediction[0]).numpy()
        
        # اختيار الفئة الأعلى ثقة
        class_idx = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100) # النتيجة مستحيل تعدي 100%
            
        result_text = selected_config['classes'].get(class_idx, "Unknown")
        
        # تحديد لون الحالة في الواجهة
        is_safe = any(word in result_text for word in ['Normal', 'Uninfected', 'سليم'])
        
        return jsonify({
            'result': result_text,
            'confidence': f"{confidence:.2f}%",
            'is_danger': 0 if is_safe else 1
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    app.run(debug=True)
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
