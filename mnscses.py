from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from comtypes import CoInitialize, CoUninitialize
from weather import getDataWeather
from threading import Lock
from threading import Timer
app = Flask(__name__)

model = YOLO("templates/models/best1.pt")
def get_camera_indexes():
    CoInitialize()
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = []
        graph = FilterGraph()
        for index, device_name in enumerate(graph.get_input_devices()):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                devices.append(index)
                cap.release()
        return devices
    finally:
        CoUninitialize()

def control_lamp(device_index, brightness):
    print(f"Камера {device_index}: Установлена яркость лампы {brightness}%")

camera_data = {}
def process_camera_feed(device_index):
    global camera_data
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.5, classes=0)
        annotator = Annotator(frame)

        has_person = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

                has_person = True

        camera_data[device_index] = {"hasPerson": has_person}

        annotated_frame = annotator.result()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



weather_lock = Lock()
weather_data = {"isDay": True, "cloudProc": 0}

def update_weather():
    global weather_data
    city = "Курск"
    try:
        new_data = getDataWeather(city)
        with weather_lock:
            weather_data = new_data
    except Exception as e:
        print(f"Ошибка обновления погоды: {e}")



def schedule_weather_update():
    update_weather()
    Timer(600, schedule_weather_update).start()

schedule_weather_update()

@app.route('/api/weather')
def get_weather_data():
    with weather_lock:
        return weather_data

@app.route('/')
def index():
    cameras = get_camera_indexes()
    return render_template('indxes.html', cameras=cameras)

@app.route('/camera/<int:device_index>')
def camera_feed(device_index):
    return Response(process_camera_feed(device_index), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/<int:device_index>')
def get_camera_data(device_index):
    global camera_data
    data = camera_data.get(device_index, {"hasPerson": False})
    return {"device_index": device_index, **data}

@app.route('/api/lamp/<int:device_index>', methods=['POST'])
def lamp_control(device_index):
    data = request.get_json()
    brightness = data.get('brightness', 0)
    control_lamp(device_index, brightness)
    return {"status": "ok"}

if __name__ == '__main__':
    
    print("Приложение запущено. Нажмите Ctrl+C для остановки.")
    app.run(debug=True)