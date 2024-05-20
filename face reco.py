import cv2
import face_recognition

# Load a sample image and learn how to recognize it
known_image = face_recognition.load_image_file("/Users/vandanagrawal/Desktop/project/face recognition/me.jpeg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Compare the current face encoding with the known face encoding
        results = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Unknown"

        if results[0]:
            name = "Known Person"

        # Draw a box around the face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()