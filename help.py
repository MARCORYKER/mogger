from flask import Flask, request, jsonify
import cv2
import numpy as np
import dlib
import os

app = Flask(__name__)

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(predictor_path)
    print("Predictor loaded successfully")
except Exception as e:
    print(f"Failed to load predictor: {e}")

def calculate_symmetry(landmarks):
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose_tip = landmarks[30]
    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_to_mid = np.linalg.norm(nose_tip - (left_eye + right_eye) / 2)
    diff = eye_distance / nose_to_mid if nose_to_mid != 0 else 1
    score = 10 - min(diff * 2, 9)
    final_score = max(min(score, 10), 1)
    print(f"Symmetry Diff: {diff}, Raw Score: {score}, Clamped: {final_score}")
    return final_score

def calculate_jawline_sharpness(landmarks):
    jawline = landmarks[0:17]
    angles = []
    for i in range(len(jawline) - 2):
        vec1 = jawline[i + 1] - jawline[i]
        vec2 = jawline[i + 2] - jawline[i + 1]
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        angles.append(angle)
    jaw_score = min(np.mean(angles) * 10, 10)
    return jaw_score

def calculate_skin_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
    skin_score = min(contrast / 50, 10)
    return skin_score

def calculate_facial_ratios(landmarks):
    width = np.linalg.norm(landmarks[0] - landmarks[16])
    height = np.linalg.norm(landmarks[8] - np.mean(landmarks[19:25], axis=0))
    ratio = height / width
    ratio_score = min(ratio * 10, 10)
    return ratio_score

def analyze_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Invalid image file"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return {"error": "No face detected"}

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        symmetry_score = calculate_symmetry(landmarks)
        jawline_score = calculate_jawline_sharpness(landmarks)
        skin_quality_score = calculate_skin_quality(img)
        facial_ratios_score = calculate_facial_ratios(landmarks)
        final_score = (symmetry_score + jawline_score + skin_quality_score + facial_ratios_score) / 4

        funny_ratings = ['Goblin Mode', 'Decent', 'Certified Mogger', 'Greek God']
        rating_index = min(int(final_score * 4 // 10), len(funny_ratings) - 1)
        rating = funny_ratings[rating_index]

        recommendations = []
        if symmetry_score < 4:
            recommendations.append(f"Symmetry at {symmetry_score:.2f} - facial exercises could help balance things out.")
        elif symmetry_score < 7:
            recommendations.append(f"Symmetry at {symmetry_score:.2f} - pretty balanced, maybe tweak posture.")
        else:
            recommendations.append(f"Symmetry at {symmetry_score:.2f} - damn, that’s symmetrical!")

        if jawline_score < 3:
            recommendations.append(f"Jawline at {jawline_score:.2f} - chew some gum to sharpen it up.")
        elif jawline_score < 6:
            recommendations.append(f"Jawline at {jawline_score:.2f} - decent edge, try mewing for more definition.")
        else:
            recommendations.append(f"Jawline at {jawline_score:.2f} - chiseled vibes, keep it up!")

        if skin_quality_score < 3:
            recommendations.append(f"Skin at {skin_quality_score:.2f} - hydrate and start a basic skincare routine.")
        elif skin_quality_score < 6:
            recommendations.append(f"Skin at {skin_quality_score:.2f} - not bad, add a cleanser to glow up.")
        else:
            recommendations.append(f"Skin at {skin_quality_score:.2f} - smooth as hell, maintain that!")

        if facial_ratios_score < 7:
            recommendations.append(f"Ratios at {facial_ratios_score:.2f} - hairstyle tweak could balance it.")
        else:
            recommendations.append(f"Ratios at {facial_ratios_score:.2f} - proportions on point!")

        if final_score > 8:
            recommendations.append("Overall mogger energy - keep slaying!")
        elif final_score < 4:
            recommendations.append("Rough start, but small tweaks can level you up!")

        return {
            "symmetry": round(symmetry_score, 2),
            "jawline": round(jawline_score, 2),
            "skinQuality": round(skin_quality_score, 2),
            "facialRatios": round(facial_ratios_score, 2),
            "finalScore": round(final_score, 2),
            "rating": rating,
            "recommendations": recommendations
        }

@app.route('/')
def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mog-o-Tron 3000</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Orbitron', sans-serif;
            }
            body {
                background: linear-gradient(135deg, #0d0d0d, #1a1a1a);
                color: #fff;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow-x: hidden;
            }
            .container {
                background: rgba(20, 20, 20, 0.95);
                border-radius: 15px;
                padding: 30px;
                width: 90%;
                max-width: 600px;
                box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
                border: 1px solid rgba(0, 255, 255, 0.3);
                animation: fadeIn 0.5s ease-in-out;
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                color: #00ffff;
                text-shadow: 0 0 10px #00ffff;
                margin-bottom: 20px;
            }
            .upload-area {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }
            input[type="file"] {
                display: none;
            }
            label {
                background: linear-gradient(45deg, #00ffff, #00b7b7);
                padding: 12px 25px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1.1em;
                transition: all 0.3s ease;
            }
            label:hover {
                transform: scale(1.05);
                box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            }
            button {
                background: linear-gradient(45deg, #ff00ff, #b700b7);
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                color: #fff;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            button:hover {
                transform: scale(1.05);
                box-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
            }
            #image-preview {
                max-width: 100%;
                margin-top: 20px;
                border-radius: 10px;
                display: none;
                border: 2px solid #00ffff;
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            }
            #results {
                margin-top: 30px;
                padding: 20px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: slideUp 0.5s ease-in-out;
            }
            #results h3 {
                color: #00ffff;
                font-size: 1.8em;
                margin-bottom: 15px;
                text-shadow: 0 0 5px #00ffff;
            }
            #results p {
                font-size: 1.2em;
                margin: 10px 0;
                color: #e0e0e0;
            }
            #results p strong {
                color: #ff00ff;
            }
            #results h4 {
                color: #00ffff;
                margin: 15px 0 10px;
                font-size: 1.4em;
            }
            #results ul {
                list-style: none;
                padding-left: 0;
            }
            #results li {
                font-size: 1.1em;
                margin: 8px 0;
                padding-left: 20px;
                position: relative;
                color: #d0d0d0;
            }
            #results li:before {
                content: "►";
                color: #00ffff;
                position: absolute;
                left: 0;
                font-size: 0.8em;
                top: 4px;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideUp {
                from { transform: translateY(20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            @media (max-width: 480px) {
                h1 { font-size: 2em; }
                .container { padding: 20px; }
                label, button { font-size: 1em; padding: 10px 20px; }
                #results p { font-size: 1em; }
                #results li { font-size: 0.9em; }
            }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    </head>
    <body>
        <div class="container">
            <h1>Mog-o-Tron</h1>
            <div class="upload-area">
                <label for="imageInput">Select Your Face</label>
                <input type="file" id="imageInput" accept="image/*">
                <button onclick="uploadImage()">Analyze</button>
            </div>
            <img id="image-preview" src="">
            <div id="results"></div>
        </div>

        <script>
            function uploadImage() {
                const input = document.getElementById("imageInput");
                if (!input.files.length) {
                    alert("Select a face to mog-ify!");
                    return;
                }

                const file = input.files[0];
                const formData = new FormData();
                formData.append("image", file);

                const preview = document.getElementById("image-preview");
                preview.src = URL.createObjectURL(file);
                preview.style.display = "block";

                fetch("/analyze", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById("results").innerHTML = `<p style="color:#ff5555;">${data.error}</p>`;
                        return;
                    }

                    document.getElementById("results").innerHTML = `
                        <h3>Analysis Complete</h3>
                        <p><strong>Symmetry:</strong> ${data.symmetry}/10</p>
                        <p><strong>Jawline:</strong> ${data.jawline}/10</p>
                        <p><strong>Skin:</strong> ${data.skinQuality}/10</p>
                        <p><strong>Ratios:</strong> ${data.facialRatios}/10</p>
                        <p><strong>Final Score:</strong> ${data.finalScore}/10</p>
                        <p><strong>Rating:</strong> ${data.rating}</p>
                        <h4>Upgrade Path</h4>
                        <ul>${data.recommendations.map(r => `<li>${r}</li>`).join("")}</ul>
                    `;
                })
                .catch(error => console.error("Error:", error));
            }
        </script>
    </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
    file = request.files['image']
    file_path = f"temp_{os.urandom(8).hex()}.jpg"
    file.save(file_path)
    result = analyze_face(file_path)
    os.remove(file_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)