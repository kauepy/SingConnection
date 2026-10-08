"""
Script de diagnóstico da webcam - roda localmente (janela OpenCV, fora do
navegador) e desenha rosto, pose e mãos detectados pelo MediaPipe Holistic.

Serve pra testar se a câmera e a detecção estão funcionando bem antes de
coletar dados. Não faz parte do pipeline de produção (coleta usa só
mp_hands, e o reconhecimento de verdade acontece no main.py via frontend).
"""

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def testar_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    print("Webcam aberta! Pressione Q para sair")

    with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = holistic.process(rgb)

            # Desenha o rosto (só se detectado)
            if resultado.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    resultado.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1),
                )

            # Desenha o corpo / pose (só se detectado)
            if resultado.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    resultado.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

            # Desenha mão esquerda (só se detectada)
            if resultado.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    resultado.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

            # Desenha mão direita (só se detectada)
            if resultado.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    resultado.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

            # Status na tela
            maos = sum([
                resultado.left_hand_landmarks is not None,
                resultado.right_hand_landmarks is not None,
            ])
            rosto = "Sim" if resultado.face_landmarks else "Não"
            corpo = "Sim" if resultado.pose_landmarks else "Não"

            cv2.putText(frame, f"Maos: {maos} | Rosto: {rosto} | Corpo: {corpo}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Libras - Holistic", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    testar_webcam()