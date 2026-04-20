import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        
        # calculate focal length
        fov_rad = np.deg2rad(fov)
        f = self.width / (2 * np.tan(fov_rad / 2))
        # principal point
        cx, cy = self.width / 2, self.height / 2

        # camera positions
        cam1_pos = np.array([0.0, 1.0, 0.0])
        cam2_pos = np.array([0.0, 2.5, 0.0])

        # BEV camera's rotation matrix
        angle = -(np.pi / 2)
        R_bev = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])

        new_pixels = []

        for p in points:
            u, v = p[0], p[1]

            # unproject BEV pixel to camera-space ray
            ray_cam = np.array([(u - cx) / f, (v - cy) / f, -1.0])
            # rotate ray into world space
            ray_world = R_bev @ ray_cam

            # skip if parallel
            if abs(ray_world[1]) < 1e-6:
                continue

            # where the ray from cam2_pos hits the ground
            t = -cam2_pos[1] / ray_world[1]
            # skip if behind camera
            if t < 0:
                continue

            world_pt = cam2_pos + t * ray_world

            # transform world point into front camera space
            p_cam1 = world_pt - cam1_pos

            # skip if behind camera
            if p_cam1[2] <= 0:
                continue

            # project onto front image
            u_front = int(round((f * p_cam1[0] / p_cam1[2]) + cx))
            v_front = int(round((f * (-p_cam1[1]) / p_cam1[2]) + cy))

            # If the projected points are out of bound, set to bound
            u_front = max(0, min(u_front, self.width - 1))
            v_front = max(0, min(v_front, self.height - 1))
            
            new_pixels.append([u_front, v_front])
        
        return np.array(new_pixels, dtype=np.int32).reshape((-1, 1, 2))

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)