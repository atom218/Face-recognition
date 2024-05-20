import cv2
import face_recognition
import mysql.connector
from datetime import datetime

frame_width = 320
frame_height = 240


video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


known_image1 = face_recognition.load_image_file("enter your path1")
known_image2 = face_recognition.load_image_file("enter your path2")
known_image3 = face_recognition.load_image_file("enter your path3")
# you can add as many as you want

known_encoding1 = face_recognition.face_encodings(known_image1)[0]
known_encoding2 = face_recognition.face_encodings(known_image2)[0]
known_encoding3 = face_recognition.face_encodings(known_image3)[0]


known_encodings = [
    known_encoding1,
    known_encoding2,
    known_encoding3
]

known_names = [
    "name label 1",
    "name label 2",
    "name label 3"
]


try:
    db = mysql.connector.connect(
        host="localhost",
        user="your mysql username",
        password="your mysql password",
        database="face_recognition_db" #this is the name of the table 
    )
    cursor = db.cursor()
    print("Connected to the database successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)


cursor.execute("TRUNCATE TABLE people_in_frame") #it clears the table before every rerun of the code. this can be removed if you want 
db.commit()
print("Table cleared.")

while True:
    
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture video frame.")
        continue

   
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown" 

       
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_names[best_match_index]

        print(f"Detected: {name}")  

        
        cursor.execute("SELECT COUNT(*) FROM people_in_frame WHERE name = %s", (name,))
        result = cursor.fetchone()

        if result[0] == 0:  
            try:
                cursor.execute("INSERT INTO people_in_frame (name) VALUES (%s)", (name,))
                db.commit() #this is used to log the data into the table 
                print(f"Logged to database: {name}") 
            except mysql.connector.Error as err:
                print(f"Error: {err}")  

        
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
      
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow('Video', frame)

    
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q on your keyboard to exit the code
        break


video_capture.release()
cv2.destroyAllWindows()


cursor.close()
db.close()
print("Database connection closed.")
