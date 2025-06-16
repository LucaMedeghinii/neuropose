import os
import cv2
import numpy as np
from datetime import datetime
import base64
from flask import Flask, render_template, Response, request, url_for, redirect, flash, session, jsonify, send_from_directory

import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'neurosecret'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inizializzazione dei moduli MediaPipe per postura, mesh facciale e mani
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Nomi delle dita in corrispondenza dei landmark principali di MediaPipe Hands
HAND_LANDMARK_NAMES = [
    "Pollice",            # 4 
    "Indice",             # 8 
    "Medio",              # 12 
    "Anulare",            # 16 
    "Mignolo"             # 20 
]

# Variabile globale per feedback sulla postura
last_posture_feedback = {"feedback": "Attendi analisi..."}

# Funzione per ottenere la cartella utente
def user_folder():
    user_id = session.get('user_id') # Prende l'id utente dalla sessione
    if not user_id:
        # Se non esiste, lo crea con timestamp e lo salva in sessione
        user_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        session['user_id'] = user_id
    folder = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(folder, exist_ok=True) # Crea la cartella utente se non esiste
    return folder

# Funzione per controllare l'allineamento delle spalle e l'inclinazione della testa
def check_shoulders_and_head(landmarks, image_width, image_height, prev_state=None):
     # Landmark per spalle e orecchie sinistra e destra
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_ear = landmarks[7]
    right_ear = landmarks[8]

     # Calcola posizione pixel delle spalle
    x_left = left_shoulder.x * image_width
    y_left = left_shoulder.y * image_height
    x_right = right_shoulder.x * image_width
    y_right = right_shoulder.y * image_height

     # Differenza verticale fra spalle (per controllare l'allineamento)
    diff_y = y_left - y_right
    threshold_y = 25 # Soglia tolleranza spalle
    shoulders_ok = abs(diff_y) <= threshold_y # True se spalle allineate

    # Differenza verticale fra orecchie (per controllare inclinazione testa)
    y_left_ear = left_ear.y * image_height
    y_right_ear = right_ear.y * image_height
    diff_ear_y = y_left_ear - y_right_ear
    threshold_ear = 20
    head_tilt_ok = abs(diff_ear_y) <= threshold_ear # True se testa non inclinata

    # Colore verde se postura corretta, rosso se errata
    if shoulders_ok and head_tilt_ok:
        color = (0, 220, 0)
    else:
        color = (0, 0, 255)

    #Se passato uno stato precedente, sfuma il colore
    if prev_state is not None:
        prev_color = prev_state.get('color', color)
        color = tuple(int((c + pc) / 2) for c, pc in zip(color, prev_color))

    new_state = {'color': color}

    return {
        'shoulders_ok': shoulders_ok,
        'head_tilt_ok': head_tilt_ok,
        'left_shoulder_pos': (int(x_left), int(y_left)),
        'right_shoulder_pos': (int(x_right), int(y_right)),
        'color': color,
        'new_state': new_state
    }

# Controlla se la schiena è dritta confrontando la posizione media delle spalle e dei fianchi
def check_back_posture(landmarks, image_width, image_height):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    
    # Calcola i punti medi in pixel tra spalle e fianchi
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2 * image_width
    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2 * image_height
    hip_mid_x = (left_hip.x + right_hip.x) / 2 * image_width
    hip_mid_y = (left_hip.y + right_hip.y) / 2 * image_height

    # Differenza orizzontale tra spalle e fianchi
    horizontal_deviation = abs(shoulder_mid_x - hip_mid_x)
    threshold_deviation = 20 # Soglia tolleranza deviazione
    back_straight = horizontal_deviation <= threshold_deviation # True se schiena dritta
    return back_straight, horizontal_deviation, (int(shoulder_mid_x), int(shoulder_mid_y)), (int(hip_mid_x), int(hip_mid_y))

# Genera il messaggio di feedback in base alla postura rilevata
def posture_feedback(shoulders_ok, head_tilt_ok, back_straight, deviation):
    if shoulders_ok and head_tilt_ok and back_straight:
        return "Postura corretta! Continua così."
    messages = []
    if not shoulders_ok:
        messages.append("Le spalle non sono allineate, cerca di mantenerle alla stessa altezza.")
    if not head_tilt_ok:
        messages.append("La testa è inclinata, prova a mantenerla dritta.")
    if not back_straight:
        messages.append("La schiena è storta, cerca di raddrizzarla mantenendo la colonna verticale.")
    return ' '.join(messages)

# Indici dei landmark MediaPipe per la testa usati nell'orientamento del volto
NOSE_TIP_IDX = 1
LEFT_EAR_IDX = 234
RIGHT_EAR_IDX = 454
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# Analizza i landmark facciali per determinare se la testa è inclinata o ruotata
def detect_head_orientation(landmarks, image_width, image_height):
    nose = landmarks[NOSE_TIP_IDX]
    left_ear = landmarks[LEFT_EAR_IDX]
    right_ear = landmarks[RIGHT_EAR_IDX]
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    chin = landmarks[152]  # mento

    # Coordinate in pixel di punti chiave
    nose_x, nose_y = nose.x * image_width, nose.y * image_height
    chin_y = chin.y * image_height
    eye_mid_y = ((left_eye.y + right_eye.y) / 2) * image_height
    ear_mid_x = ((left_ear.x + right_ear.x) / 2) * image_width

    horiz_diff = nose_x - ear_mid_x # Differenza orizzontale tra naso e centro orecchie
    vertical_chin_diff = chin_y - nose_y # Differenza verticale tra mento e naso

     # Soglie proporzionali alla dimensione immagine
    horiz_thresh = image_width * 0.03
    chin_high_thresh = image_height * 0.13
    chin_low_thresh = image_height * 0.07

    orientation = []

    # Rotazione orizzontale
    if horiz_diff > horiz_thresh:
        orientation.append('Testa girata a sinistra')
    elif horiz_diff < -horiz_thresh:
        orientation.append('Testa girata a destra')

    # Inclinazione verticale
    if vertical_chin_diff > chin_high_thresh:
        orientation.append('Testa alzata')
    elif vertical_chin_diff < chin_low_thresh:
        orientation.append('Testa abbassata')

    if not orientation:
        return 'Sguardo dritto.'
    else:
        return '{}. Mantieni lo sguardo dritto.'.format(', '.join(orientation))

# ---- ROUTES FLASK ----

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Pagina per la visualizzazione della posa 3D
@app.route('/3dpose')
def pose3d():
    return render_template('3dpose.html')

# Pagina per la webcam (modalità selezionabile: pose, face, hands)
@app.route('/webcam', methods=['GET'])
def webcam():
    mode = request.values.get('mode', 'pose')
    show_face = True
    return render_template('webcam.html',
                           mode=mode,
                           show_face=show_face)

# Endpoint per lo streaming video della webcam
@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'pose')
    show_face = request.args.get('show_face', '1') == '1'  # Restituisce una risposta HTTP multipart contenente i frame elaborati
    return Response(gen_frames(mode, show_face),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Generatore di frame per lo streaming video
def gen_frames(mode, show_face):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: webcam non disponibile")
        return

    # Inizializza i modelli MediaPipe
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_feedback_state = None # Stato precedente per evitare feedback ripetitivi
    global last_posture_feedback # Variabile globale per salvare l’ultimo messaggio di feedback

    while True:
        success, frame = cap.read() # Legge un frame dalla videocamera
        if not success:
            break
        frame = cv2.resize(frame, (640, 360)) # Ridimensiona il frame per uniformità
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte il frame da BGR a RGB
        image.flags.writeable = False # Rende l’immagine non modificabile per aumentare la performance dei modelli

        results_pose = None
        results_face = None
        results_hands = None

        # Selezione modalità di analisi
        if mode == 'pose':
            results_pose = pose.process(image)
        elif mode == 'face':
            results_face = face_mesh.process(image)
        elif mode == 'hands':
            results_hands = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Analisi della postura
        if mode == 'pose' and results_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Disegna i landmarks del corpo
            landmarks = results_pose.pose_landmarks.landmark # Estrae i punti rilevati

            feedback = check_shoulders_and_head(
                landmarks, image.shape[1], image.shape[0], prev_feedback_state) # Verifica la posizione delle spalle e testa
            prev_feedback_state = feedback['new_state'] # Aggiorna lo stato del feedback

            back_straight, deviation, shoulder_mid, hip_mid = check_back_posture(
                landmarks, image.shape[1], image.shape[0]) # Controlla se la schiena è dritta
            cv2.circle(image, feedback['left_shoulder_pos'], 18, feedback['color'], -1, lineType=cv2.LINE_AA)
            cv2.circle(image, feedback['right_shoulder_pos'], 18, feedback['color'], -1, lineType=cv2.LINE_AA)
            color_line = (0, 220, 0) if back_straight else (0, 0, 255)
            cv2.line(image, shoulder_mid, hip_mid, color_line, 4)

            # Genera un messaggio di feedback basato sull'analisi
            feedback_message = posture_feedback(feedback['shoulders_ok'], feedback['head_tilt_ok'], back_straight, deviation)
            last_posture_feedback["feedback"] = feedback_message # Salva il feedback

        # Analisi del volto
        elif mode == 'face' and results_face and results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)) # Disegna la mesh facciale
                orientation_msg = detect_head_orientation(face_landmarks.landmark, image.shape[1], image.shape[0]) # Calcola l'orientamento della testa
                last_posture_feedback['feedback'] = orientation_msg # Salva il feedback dell’orientamento

        # Analisi delle mani
        elif mode == 'hands' and results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Disegna i landmark delle mani
                h, w = image.shape[0], image.shape[1]
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)  # Converte le coordinate normalizzate in pixel
                    cv2.circle(image, (x, y), 6, (0, 0, 255), -1)
                    # Mostra il nome sulle punte delle dita principali
                    tip_indices = [4, 8, 12, 16, 20]
                    if idx in tip_indices:
                        name = HAND_LANDMARK_NAMES[tip_indices.index(idx)]
                        cv2.putText(image, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                        last_posture_feedback["feedback"] = "Mano rilevata: nome di ogni dito sulle punte."
        else:
            # Nessun risultato rilevato
            if mode == 'pose':
                last_posture_feedback["feedback"] = "Postura non rilevata"
            elif mode == 'face':
                last_posture_feedback["feedback"] = "Volto non rilevato"
            elif mode == 'hands':
                last_posture_feedback["feedback"] = "Mano non rilevata"
            else:
                last_posture_feedback["feedback"] = "Modalità non riconosciuta"

        # Codifica il frame in JPEG e lo invia come stream
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            break
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Rilascio delle risorse
    cap.release()
    pose.close()
    face_mesh.close()
    hands.close()

# Endpoint API per ottenere il feedback corrente sulla postura
@app.route('/posture_feedback')
def posture_feedback_api():
    return jsonify(last_posture_feedback)

# Salva uno screenshot inviato dal client (base64) nella cartella utente
@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    data = request.get_json()
    image_data = data.get('image')
    mode = data.get('mode', 'pose')
    if not image_data:
        return jsonify({'message': 'Nessuna immagine ricevuta'}), 400

    if ',' in image_data:
        image_base64 = image_data.split(',')[1]
    else:
        image_base64 = image_data

    folder = user_folder()
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(folder, filename)

    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(image_base64))

    return jsonify({'message': f'Screenshot salvato come {filename}'}), 200

# Visualizza la galleria degli screenshot dell'utente
@app.route('/gallery')
def gallery():
    folder = user_folder()
    files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')], reverse=True)
    return render_template('gallery.html', images=files, user_id=session.get('user_id'))

# Elimina uno screenshot selezionato dalla galleria
@app.route('/delete_screenshot', methods=['POST'])
def delete_screenshot():
    filename = request.form.get('filename')
    folder = user_folder()
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for('gallery'))

# Serve i file statici caricati (screenshot)
@app.route('/uploads/<user_id>/<filename>')
def uploaded_file(user_id, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, user_id), filename)

# Pagina per il caricamento di immagini statiche e la loro analisi
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result_img = None
    posture_message = None
    mode = request.form.get('mode', 'pose')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nessun file selezionato')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Nessun file selezionato')
            return redirect(request.url)
        if file:
            filename = 'upload_input.jpg'
            filepath = os.path.join('static', filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            if image is None:
                flash('Errore nel caricamento immagine')
                return redirect(request.url)
            # Analisi postura statica
            if mode == 'pose':
                with mp_pose.Pose(static_image_mode=True) as pose:
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        posture_message = "Analisi completata!"
                    else:
                        posture_message = "Nessuna postura rilevata."
            # Analisi volto statico
            elif mode == 'face':
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
                        posture_message = "Volto rilevato!"
                    else:
                        posture_message = "Nessun volto rilevato."
            # Analisi mani statiche
            elif mode == 'hands':
                with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            h, w = image.shape[0], image.shape[1]
                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                x, y = int(landmark.x * w), int(landmark.y * h)
                                cv2.circle(image, (x, y), 6, (0, 0, 255), -1)
                                if idx in [4, 8, 12, 16, 20]:
                                    name = HAND_LANDMARK_NAMES[idx]
                                    cv2.putText(image, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                        posture_message = "Mano rilevata: nome di ogni dito sulle punte."
                    else:
                        posture_message = "Nessuna mano rilevata."
            outname = 'upload_result.jpg'
            outpath = os.path.join('static', outname)
            cv2.imwrite(outpath, image)
            result_img = outname

    return render_template('upload.html',
                           result_img=result_img,
                           posture_message=posture_message,
                           mode=mode)

# Avvio dell'app Flask in modalità debug
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
