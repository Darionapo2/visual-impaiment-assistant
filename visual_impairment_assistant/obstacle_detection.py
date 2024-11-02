import numpy as np
import cv2
from Obstacle import *
from PIL import Image, ImageFont, ImageDraw

obstacles = []

def detect(frame, previous_obstacles, queue_rgb):
    mask_obstacle_filter = cv2.inRange(frame, 400, 1500)
    closed_mask = cv2.morphologyEx(mask_obstacle_filter, cv2.MORPH_CLOSE, kernel = np.ones((7, 7)))

    closed_mask_rgb = cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2RGB)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = 1
    sorted_contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    filtered_contours = sorted_contours[:n]

    new_centers = []
    for contour in filtered_contours:

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            detected_center = np.array((cx, cy))
            new_centers.append(detected_center)

            found = False
            for o in previous_obstacles:
                if o.is_near(detected_center, 50):
                    o.distance = o.calculate_distance(frame, contour)
                    o.center = detected_center
                    o.contour = contour
                    found = True
                    break

                elif o.is_near(detected_center, o.radius):
                    o.area += cv2.contourArea(contour)
                    o.update_radius()
                    o.distance = o.calculate_distance(frame, contour)
                    o.contour = contour
                    found = True
                    break

            if not found:
                obstacles.append(Obstacle(cv2.contourArea(contour), detected_center, frame, contour, use_min_distance = True))

    clean_obstacles = [o for o in obstacles for center in new_centers if o.is_near(center, 50)]
    print(clean_obstacles)

    new_obstacles = list(set.intersection(set(obstacles), set(clean_obstacles)))

    for o in new_obstacles:
        o.text_location = utils.locate_obstacles(1280, 720, np.array([0.34, 0.33]), np.array([0.5]), o)

    # contour_video_writer.release()
    # binary_video_writer.release()


    return new_obstacles, closed_mask_rgb, filtered_contours

'''
def draw_obstacles(frame, obstacles):
    for o in obstacles:
        cv2.circle(frame, o.center, 5, (255, 0, 255), 5)

        cv2.circle(frame, o.center, int(o.radius), (255, 0, 0), 3)

        cv2.putText(frame, text = o.text_location, org = o.center,
                    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.2, color = (255, 255, 0),
                    thickness = 1, lineType = cv2.LINE_AA)
'''
def draw_obstacles(frame, obstacles):
    # Disegna i cerchi sugli ostacoli con OpenCV
    for o in obstacles:
        cv2.circle(frame, o.center, 5, (0, 0, 255), 5)
        cv2.circle(frame, o.center, int(o.radius), (0, 0, 255), 3)

    # Converti il frame di OpenCV in un'immagine PIL (dopo aver disegnato i cerchi)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Carica il font Arial
    font = ImageFont.truetype('consolab.ttf', 45)  # Assicurati che il percorso del font sia corretto

    for o in obstacles:
        # Controlla le coordinate del testo
        print(f"Drawing text at position: {o.center}")

        # Aggiungi il testo utilizzando Pillow (su pil_image)
        text = o.text_location
        draw.text((o.center[0], o.center[1]), text, font=font, fill=(255, 0, 0))  # Colore rosso (RGB)

    # Converti nuovamente l'immagine PIL in formato OpenCV
    result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return result_frame


def join_non_null_values(frames):
    joined_frame = np.zeros_like(frames[0], dtype = frames[0].dtype)

    for frame in frames:
        mask = frame != 0
        joined_frame = np.where(mask, frame, joined_frame)

    return joined_frame

def show_obstacles(queue_depth, queue_obstacles, queue_rgb, stop_event):
    # obs_video_writer = cv2.VideoWriter('obs.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (1280, 720))
    # binary_video_writer = cv2.VideoWriter('bin.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (1280, 720))
    # contour_video_writer = cv2.VideoWriter('border.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (1280, 720))

    obstacles_list = []
    # k = 0
    # j = 0

    loops = 300
    while not stop_event.is_set():
        if queue_depth.qsize() >= 3:

            raw_frames = []
            for i in range(3):
                raw_frames.append(queue_depth.get())

            if queue_rgb.qsize() >= 1:
                frame_rgb = queue_rgb.get()

            joined_frame = join_non_null_values(raw_frames)

            obstacles_list, bin_mask, contours = detect(joined_frame, obstacles_list, queue_rgb)

            if not queue_obstacles.full():
                queue_obstacles.put(obstacles_list)

            cv2.drawContours(frame_rgb, contours, -1, (0, 255, 0), 3)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(joined_frame, alpha = 0.03), cv2.COLORMAP_JET)
            drawn_frame = draw_obstacles(depth_colormap, obstacles_list)

            cv2.imshow('Depth Frame combined', drawn_frame)
            cv2.imshow('bin mask', bin_mask)
            cv2.imshow('borders', frame_rgb)


            # obs_video_writer.write(drawn_frame)

            # binary_video_writer.write(bin_mask)

            # contour_video_writer.write(frame_rgb)

            '''
            k += 1
            print(k)
            if k % 60 == 0:
                j += 1
                cv2.imwrite(f'obs_frame{j}.jpg', depth_colormap)
                print('I just saved a depth frame') 
            '''

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            loops -= 1
            print(loops)
            if loops == 0:
                break


    print('ho fatto un release')
    # obs_video_writer.release()
    # contour_video_writer.release()
    # binary_video_writer.release()
    cv2.destroyAllWindows()