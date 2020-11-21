from cv2 import cv2
import numpy as np 
import sys
import face_recognition

# ## ACTIONS
# Example: Text yourself when someone has been recognized by the camera
# import smtplib

FILE_PATH = "./photos/"

#change to name of face image you placaed in photos folder
PHOTO_FILE = "someone.jpg"
# FRIEND_FILE = "friend.jpg"

your_image = face_recognition.load_image_file(FILE_PATH + PHOTO_FILE)
your_face_encoding = face_recognition.face_encodings(your_image)[0]

# CAN DO MULTIPLE
# friend_image = face_recognition.load_image_file(FRIEND_FILE)
# friend_face_encoding = face_recognition.face_encodings(friend_image)[0]

known_face_encodings = [
    your_face_encoding,
    #friend_face_encoding
]

known_face_names = [
    "Person Name",
    # "Friend Name"
]

face_locations = []
face_encodings = []
face_names = []
DO_TASK= 0
process_this_frame = True

cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('outpy2.mp4',fourcc, 15, (640,480))
while True:
    if DO_TASK == 1:
        DO_TASK += 1
        #### Do something now that the face has been recognized
        # EX includes using email to text a phone number when a face was recognized
        # server = smtplib.SMTP( "smtp.gmail.com", 587 )
        # server.starttls()
        # server.login( 'email', 'password' )
        # server.sendmail( 'from', '1234567890@txt.att.net', 'Persons Face Recognized' )

        print("Finished Task")
    ret, frame = cap.read()

    # If ret is false that means there is no webcam / capture was lost, therefore cannot process any frames and will exit the program
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]

    
    # out.write(frame)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Label faces as unknown initially
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                print("Face Recognized")
                ## Find a better way to tell program to do the task
                DO_TASK += 1

                # Add label to face in frame if known
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        left *= 4
        bottom *= 4

        cv2.rectangle(frame, (left,top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# releases cv2 and
cap.release()
# out.release()


cv2.destroyAllWindows()