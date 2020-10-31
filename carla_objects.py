import carla
import numpy as np
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE, K_p
import time


class World(object):
    def __init__(self,
                 client: carla.Client,
                 map_name: str = None,
                 screen_size=(640, 480),
                 camera_pos=(-80, -10, 80),
                 camera_rot=(0.0, 0.0, 0.0)):
        self.client: carla.Client = client
        self.carla_world: carla.World = client.get_world()
        self.screen_size = screen_size
        self.map: carla.Map = self.carla_world.get_map()
        # for map_name in self.client.get_available_maps():
        #     print(map_name)

        if map_name is not None and self.map.name != map_name:
            print("Loading", map_name, "world...")
            self.carla_world = client.load_world(map_name)
            time.sleep(5.0)
            print("New map loaded.")
        self.destroy_actors()

        spectator = self.carla_world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2]),
                carla.Rotation(pitch=camera_rot[0], yaw=camera_rot[1], roll=camera_rot[2])
            )
        )

        pygame.init()
        pygame.font.init()
        # fonts = [x for x in pygame.font.get_fonts()]
        self.font = pygame.font.match_font("couriernew")
        # self.font = pygame.font.Font(font, 16)
        self.display = pygame.display.set_mode(
            screen_size,
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.clock = pygame.time.Clock()
        self.server_clock = pygame.time.Clock()
        self.is_done = False
        self.carla_world.on_tick(self.on_server_tick)

    def on_server_tick(self, timestamp):
        self.server_clock.tick()

    def draw_string(self, location: carla.Location, s: str, life_time=1.0, color=(255, 0, 0)):
        self.carla_world.debug.draw_string(
            location,
            s, draw_shadow=False,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=life_time,
            persistent_lines=True
        )

    def draw_box(self, box: carla.BoundingBox, rot: carla.Rotation, thickness=0.1, color=(255,0,0), life_time=-1.0):
        self.carla_world.debug.draw_box(
            box, rot,
            thickness=thickness,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=life_time
        )

    def render_image(self, image) -> pygame.Surface:
        if image is None:
            return None
        surface = pygame.surfarray.make_surface(image)
        self.display.blit(surface, (0, 0))
        return surface

    def render_text(self, text, pos=(0, 0), font_family="couriernew", font_size=16, color=(255, 255, 255), bold=False, italic=False):
        font = pygame.font.SysFont(font_family, font_size, bold=bold, italic=italic)
        # font = pygame.font.Font(self.font, font_size, bold=bold)
        self.display.blit(font.render(text, True, color), pos)

    def redraw_display(self, fps=60):
        pygame.display.flip()
        self.clock.tick_busy_loop(fps)

    def destroy_actors(self):
        for actor in self.carla_world.get_actors().filter("vehicle.*"):
            actor.destroy()
        for actor in self.carla_world.get_actors().filter("walker.*"):
            actor.destroy()
        for actor in self.carla_world.get_actors().filter("sensor.*"):
            actor.destroy()

    def key_handler(self):
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                self.is_done = True
            elif e.type == pygame.KEYUP and e.key == K_ESCAPE:
                self.is_done = True
        return events


class Vehicle(object):
    NO_PILOT = "Manual"
    ASSIST_PILOT = "Assisted Driving"
    PID_PILOT = "PID Control"
    BC_PILOT = "Behavior Cloning"
    FOLLOW_PILOT = "Follow"
    DQN_PILOT = "DQN"
    DDQN_PILOT = "DDQN"

    def __init__(self,
                 world: World,
                 role_name: str,
                 color: str = "0.0, 255.0, 0.0",
                 bp_filter: str = "vehicle.*",
                 start_tf: carla.Transform = None,
                 debug: bool = False):
        self.world: World = world
        self.role_name: str = role_name
        self.color: str = color
        self.bp_filter: str = bp_filter
        self.debug = debug

        self.carla_world: carla.World = self.world.carla_world
        self.map: carla.Map = self.carla_world.get_map()
        bp_lib: carla.BlueprintLibrary = self.carla_world.get_blueprint_library()
        bps = bp_lib.filter(self.bp_filter)
        spawn_points = self.map.get_spawn_points()

        bp = np.random.choice(bps)
        bp.set_attribute("role_name", role_name)
        bp.set_attribute("color", color)
        self.start_tf: carla.Transform = np.random.choice(spawn_points)
        self.start_tf.location.z = 1.0
        if start_tf is not None:
            self.start_tf = start_tf
        self.actor: carla.Actor = self.carla_world.spawn_actor(bp, self.start_tf)

        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "640")
        bp.set_attribute("image_size_y", "480")
        bp.set_attribute("fov", "110")
        tf: carla.Transform = carla.Transform(
            carla.Location(x=2.5, y=0.0, z=1.5),
            carla.Rotation(pitch=-20.0, yaw=0.0, roll=0.0)
        )
        self.rgb_camera: carla.Actor = self.carla_world.spawn_actor(bp, tf, attach_to=self.actor)
        self.rgb_camera.listen(lambda data: self.rgb_camera_handler(data))
        self.rgb_image = None

        bp = bp_lib.find("sensor.other.collision")
        tf: carla.Transform = carla.Transform()
        self.collision_detector: carla.Actor = self.carla_world.spawn_actor(bp, tf, attach_to=self.actor)
        self.collision_detector.listen(lambda event: self.collision_handler(event))
        self.has_collided = False

        bp = bp_lib.find("sensor.other.obstacle")
        bp.set_attribute("distance", "10")
        bp.set_attribute("hit_radius", "0.5")
        bp.set_attribute("debug_linetrace", str(self.debug))
        tf: carla.Transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.0),
            carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
        )
        self.obstacle_left: carla.Actor = self.carla_world.spawn_actor(bp, tf, attach_to=self.actor)
        self.obstacle_left.listen(self.obstacle_left_handler)
        self.distance_left = 0

        bp = bp_lib.find("sensor.other.obstacle")
        bp.set_attribute("distance", "10")
        bp.set_attribute("hit_radius", "0.5")
        bp.set_attribute("debug_linetrace", str(self.debug))
        tf: carla.Transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.0),
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
        )
        self.obstacle_right: carla.Actor = self.carla_world.spawn_actor(bp, tf, attach_to=self.actor)
        self.obstacle_right.listen(self.obstacle_right_handler)
        self.distance_right = 0

        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.reverse = False
        self.auto_pilot = Vehicle.NO_PILOT

    def get_location(self) -> carla.Location:
        return self.actor.get_location()

    def get_transform(self) -> carla.Transform:
        return self.actor.get_transform()

    def obstacle_left_handler(self, data):
        self.distance_left = data.distance

    def obstacle_right_handler(self, data):
        self.distance_right = data.distance

    def rgb_camera_handler(self, data: carla.Image):
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((480, 640, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.swapaxes(0, 1)
        self.rgb_image = img

    def collision_handler(self, event: carla.CollisionEvent):
        print("Crash with", event.other_actor)
        self.has_collided = True

    def action(self):
        # print(self.throttle, self.brake, self.steer, self.reverse)
        self.control(self.throttle, self.brake, self.steer, self.reverse)

    def control(self,
                throttle: float = 0.0,
                brake: float = 0.0,
                steer: float = 0.0,
                reverse: bool = False):
        self.actor.apply_control(
            carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=float(steer),
                reverse=bool(reverse)
            )
        )

    def speed_mps(self) -> float:
        v: carla.Vector3D = self.actor.get_velocity()
        s = np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        return s

    def speed_kmh(self) -> float:
        return 3.6 * self.speed_mps()

    def destroy(self):
        self.actor.destroy()
        self.rgb_camera.destroy()
        self.collision_detector.destroy()
        self.obstacle_left.destroy()
        self.obstacle_right.destroy()

    def key_handler(self, events):
        for e in events:
            if e.type == pygame.KEYUP:
                if e.key == K_w:
                    self.throttle = 0.0
                elif e.key == K_s:
                    self.reverse = False
                    self.throttle = 0.0
                elif e.key == K_a:
                    self.steer = 0.0
                elif e.key == K_d:
                    self.steer = 0.0
                elif e.key == K_SPACE:
                    self.brake = 0.0
                elif e.key == K_p:
                    if self.auto_pilot == Vehicle.NO_PILOT:
                        self.auto_pilot = Vehicle.PID_PILOT
                    else:
                        self.auto_pilot = Vehicle.NO_PILOT
            elif e.type == pygame.KEYDOWN:
                if e.key == K_w:
                    self.throttle = 1.0
                elif e.key == K_s:
                    self.reverse = True
                    self.throttle = 1.0
                elif e.key == K_a:
                    self.steer = -1.0
                elif e.key == K_d:
                    self.steer = 1.0
                elif e.key == K_SPACE:
                    self.brake = 1.0
